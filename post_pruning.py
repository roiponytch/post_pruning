
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import resnet
import resnet
from utils import *
import time
os.environ['WANDB_CONSOLE'] = 'off'

models_map={'resnet18': resnet.resnet18,
            'resnet34': resnet.resnet34,
            'resnet50': resnet.resnet50}

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device, dtype=torch.float):
        self.dl = dl
        self.device = device
        self.dtype= dtype
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device, self.dtype)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def calc_conv_filter_ratio(clamp_th=0.1):
    valid_filters =[]
    for i in range(len(resnet.global_info_filt)):
        valid_filters.append(torch.sigmoid(resnet.global_info_filt[i]) >= clamp_th)
        # valid_filters.append( F.relu6(resnet.global_info_filt[i]) >= clamp_th)

        # valid_filters.append((resnet.global_info_filt[i]) >= clamp_th)

    total_conv_params=[]
    total_valid_conv_params=[]
    for i in range(len(resnet.global_info_filt)):

        if i==0:
            total_conv_params.append(3*resnet.global_info_filt[i].shape[1])
            total_valid_conv_params.append(3 * torch.sum(valid_filters[i]).cpu().numpy())
        else:
            total_conv_params.append(resnet.global_info_filt[i-1].shape[1]*resnet.global_info_filt[i].shape[1])
            total_valid_conv_params.append((torch.sum(valid_filters[i-1])*torch.sum(valid_filters[i])).cpu().numpy())

    total_conv_params = np.array(total_conv_params)
    total_valid_conv_params = np.array(total_valid_conv_params)

    return total_conv_params, total_valid_conv_params

def post_pruning(model_name,
                 train_ds,
                 valid_ds,
                 dataset_type = 'LSVRC2012',
                 num_classes=1000,
                 batch_size=64,
                 checkpoint_path=None,
                 summary_path='logs',
                 train_net_before_pruning= False,
                 fp16= False
                 ):
    device = get_default_device()
    device

    summary_path=summary_path+'/{}/{}'.format(dataset_type,model_name)
    writer = SummaryWriter(summary_path)


    model = models_map[model_name](pretrained=False,num_classes=num_classes,dataset_type=dataset_type).to(device)

    # PyTorch data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=2, pin_memory=True)

    if fp16:
        model.half()
        data_dtype = torch.half
    else:
        data_dtype = torch.float

    train_dl = DeviceDataLoader(train_dl, device, data_dtype)
    valid_dl = DeviceDataLoader(valid_dl, device, data_dtype)



    # model.half()
    # %%
    epochs = 50
    max_lr = 0.001
    grad_clip = None # 0.1
    weight_decay = 1e-5
    opt_func = torch.optim.Adam


    if train_net_before_pruning:
        history = [evaluate(model, valid_dl)]
        history
        history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                                     grad_clip=grad_clip,
                                     weight_decay=weight_decay,
                                     opt_func=opt_func)

        # save model
        torch.save({
                    'model_state_dict': model.state_dict()
                    },checkpoint_path )

    else:
        checkpoint = torch.load(checkpoint_path)
        if checkpoint.keys().__contains__('model_state_dict'):
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model(next(iter(valid_dl))[0])
        resnet.collect_info_filt=False
        history = [evaluate(model, valid_dl)]
        print(history)


    epochs = 80
    max_lr = 0.5
    grad_clip = 0.1
    weight_decay = 0.
    opt_func = torch.optim.Adam
    l1_lambda = 1. #0.001
    task_lambda = 1.
    clamp_th = 0.1

    torch.cuda.empty_cache()

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.info_filt_params, max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_dl))
    scalar = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        # Training Phase
        model.eval()
        train_losses = []
        lrs = []
        running_prunning_loss = 0.
        running_task_loss = 0.
        running_weight_loss =0.

        total_batches = len(train_dl)
        for batch in train_dl:

            images, labels = batch
            # images = images.to(device)
            # labels=labels.to(device)
            with torch.cuda.amp.autocast():
                out_student = model(images, use_info_filt=True)  # Generate predictions
                out_teacher = model(images, use_info_filt=False).detach()

                task_loss = F.cross_entropy(out_student, out_teacher.argmax(dim=1))  # Calculate loss
                pruning_loss = torch.sigmoid(torch.cat(model.info_filt_params, 1)).abs().mean()
                # pruning_loss = F.relu6(torch.cat(model.info_filt_params,1)).abs().mean()
                # pruning_loss = (torch.cat(model.info_filt_params,1)).abs().mean()
                loss = l1_lambda * pruning_loss + task_lambda * task_loss

            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()



            # task_loss = F.cross_entropy(out_student, out_teacher.argmax(dim=1))  # Calculate loss
            # pruning_loss = torch.sigmoid(torch.cat(model.info_filt_params,1)).abs().mean()
            # # pruning_loss = F.relu6(torch.cat(model.info_filt_params,1)).abs().mean()
            # # pruning_loss = (torch.cat(model.info_filt_params,1)).abs().mean()
            # loss = l1_lambda*pruning_loss + task_lambda*task_loss

            train_losses.append(loss)
            # loss.backward()

            # Gradient clipping
            # if grad_clip:
            #     nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            # optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            running_task_loss+=task_loss
            running_prunning_loss+=pruning_loss
            running_weight_loss+= loss
            lrs.append(get_lr(optimizer))
            sched.step()
        print('pruning loss: {:.4}'.format(pruning_loss))
        writer.add_scalar("Loss/pruning_loss", running_prunning_loss/total_batches, epoch)
        writer.add_scalar("Loss/running_task_loss", running_task_loss/total_batches, epoch)
        writer.add_scalar("Loss/total_loss", running_weight_loss/total_batches, epoch)
        writer.add_scalar("Loss/lambda_l1", l1_lambda, epoch)
        writer.add_scalar("Loss/lambda_task", task_lambda, epoch)
        # Validation phase
        result = evaluate(model, valid_dl)
        writer.add_scalar("Accuracy/val", result['val_acc'], epoch)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        total_conv_params, total_valid_conv_params = calc_conv_filter_ratio(clamp_th=clamp_th)
        conv_filters_left = total_valid_conv_params.sum()/total_conv_params.sum()
        print('conv filters left: {:.4}'.format(conv_filters_left))
        writer.add_scalar("conv filters left", conv_filters_left, epoch)
        model.epoch_end(epoch, result)
        history.append(result)
        writer.flush()

    writer.close()
    prune_model =model.clone()
    prune_model.load_state_dict(model.state_dict())
    prune_model = to_device(prune_model,device)
    prune_model.eval()

    total_conv_params, total_valid_conv_params = calc_conv_filter_ratio(clamp_th=clamp_th)

    conv_filters_left = total_valid_conv_params.sum() / total_conv_params.sum()
    print('conv filters left: {:.4}'.format(conv_filters_left))

    for i in range(len(prune_model.info_filt_params)):
        prune_model.info_filt_params[i].data = torch.where(torch.sigmoid(prune_model.info_filt_params[i]) < clamp_th,
                                                      torch.ones_like(prune_model.info_filt_params[i]).mul(-100.),
                                                      prune_model.info_filt_params[i])
        # valid_filters.append(torch.sigmoid(prune_model.info_filt_params[i]) >= clamp_th)


    prune_model_without_th_acc = evaluate(model, valid_dl)
    prune_model_with_th_acc = evaluate(prune_model, valid_dl)
    model.use_info_filt = False
    full_model_acc = evaluate(model, valid_dl)

    print('Full model top1 ACC: {.4f} \n'
          'Pruned model top1 ACC (w/o threshold): {.4f} \n'
          'Pruned model top1 ACC (threshold: {.2f}): {.4f}'.format(full_model_acc,
                                                             prune_model_without_th_acc,
                                                             clamp_th,
                                                             prune_model_with_th_acc))





# def plot_accuracies(history):
#     accuracies = [x['val_acc'] for x in history]
#     plt.plot(accuracies, '-x')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.title('Accuracy vs. No. of epochs');