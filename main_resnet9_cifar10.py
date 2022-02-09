
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
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import resnet_imagenet as resnet
from utils import *
import torchvision.transforms as transforms
import time
os.environ['WANDB_CONSOLE'] = 'off'

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

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

device = get_default_device()
device

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


# %%

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False, info_filt=None):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = pool
        if self.pool:
            self.pool_layer = nn.MaxPool2d(2)

        if info_filt ==None:
            self.info_filt = torch.nn.Parameter(
                torch.ones(size=[1,out_channels,1,1], requires_grad=True, device='cuda').mul(10))
        else:
            self.info_filt = info_filt

    def forward(self, xb, use_info_filt = True):
            if use_info_filt:
                out = self.conv_layer(xb)*torch.sigmoid(self.info_filt)
            else:
                out = self.conv_layer(xb)

            out = self.bnorm(out)
            out = self.relu(out)
            if self.pool:
                out = self.pool_layer(out)
            return out


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1_conv1 = conv_block(128, 128)
        self.res1_conv2 = conv_block(128, 128, info_filt=self.conv2.info_filt)

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2_conv1 = conv_block(512, 512)
        self.res2_conv2 = conv_block(512, 512,info_filt=self.conv4.info_filt)

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))


        self.info_filt_params=[]
        self.model_params = []

        for name, param in self.named_parameters():
            if name.endswith('info_filt'):
                self.info_filt_params.append(param)
            else:
                self.model_params.append(param)

        self.use_info_filt= True

    def forward(self,xb, use_info_filt =None):
        if use_info_filt == None:
            use_info_filt = self.use_info_filt

        self.conv1_out = self.conv1(xb,use_info_filt =use_info_filt )
        self.conv2_out = self.conv2(self.conv1_out,use_info_filt =use_info_filt )

        self.res1_conv1_out = self.res1_conv1(self.conv2_out, use_info_filt=use_info_filt)
        self.res1_conv2_out = self.res1_conv2(self.res1_conv1_out, use_info_filt=use_info_filt)+ self.conv2_out

        self.conv3_out = self.conv3(self.res1_conv2_out ,use_info_filt =use_info_filt )
        self.conv4_out = self.conv4(self.conv3_out,use_info_filt =use_info_filt )

        self.res2_conv1_out = self.res2_conv1(self.conv4_out, use_info_filt=use_info_filt)
        self.res2_conv2_out = self.res2_conv2(self.res2_conv1_out, use_info_filt=use_info_filt)+ self.conv4_out

        self.out = self.classifier(self.res2_conv2_out)
        return self.out



# model = to_device(ResNet9(3, 10), device)
# model = to_device(resnet.resnet18(pretrained=False,num_classes=10), device)
model = resnet.resnet18(pretrained=False,num_classes=10).to(device)
model


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.model_params, max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# %%

history = [evaluate(model, valid_dl)]
history

epochs = 3
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


if train_net_before_pruning:
    history = [evaluate(model, valid_dl)]
    history
    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                                 grad_clip=grad_clip,
                                 weight_decay=weight_decay,
                                 opt_func=opt_func)




epochs = 80
max_lr = 0.1
grad_clip = 0.1
weight_decay = 0.
opt_func = torch.optim.Adam
l1_lambda = 1. #0.001
task_lambda = 1.
clamp_th = 0.1

# checkpoint_path = 'resnet9_cifar10_trained.pth'
# checkpoint_path = 'resnet9_cifar10_trained_prune.pth'

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

torch.cuda.empty_cache()

# Set up cutom optimizer with weight decay
optimizer = opt_func(model.info_filt_params, max_lr, weight_decay=weight_decay)
# Set up one-cycle learning rate scheduler
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                            steps_per_epoch=len(train_dl))

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

        out_student = model(images,use_info_filt=True) # Generate predictions
        out_teacher = model(images, use_info_filt=False).detach()

        task_loss = F.cross_entropy(out_student, out_teacher.argmax(dim=1))  # Calculate loss
        pruning_loss = torch.sigmoid(torch.cat(model.info_filt_params,1)).abs().mean()
        loss = l1_lambda*pruning_loss + task_lambda*task_loss

        train_losses.append(loss)
        loss.backward()

        # Gradient clipping
        # if grad_clip:
        #     nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
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
prune_model =resnet.resnet34(pretrained=False,num_classes=1000).to(device)
prune_model.load_state_dict(model.state_dict())
prune_model = to_device(prune_model,device)
prune_model.eval()

total_conv_params, total_valid_conv_params = calc_conv_filter_ratio(clamp_th=clamp_th)


for i in range(len(prune_model.info_filt_params)):
    prune_model.info_filt_params[i].data = torch.where(torch.sigmoid(prune_model.info_filt_params[i]) < clamp_th,
                                                  torch.ones_like(prune_model.info_filt_params[i]).mul(-100.),
                                                  prune_model.info_filt_params[i])
    # valid_filters.append(torch.sigmoid(prune_model.info_filt_params[i]) >= clamp_th)
print(evaluate(model, valid_dl))
print(evaluate(prune_model, valid_dl))

exit()




def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');