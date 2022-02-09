
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
import resnet
from utils import *



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


# mm=resnet.resnet18(pretrained=False,num_classes=10)
train_net_before_pruning= False
writer = SummaryWriter('logs/000/')

# Look into the data directory
data_dir = '/media/Data/database/cifar10'
print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)


# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor(),
                         tt.Normalize(*stats,inplace=True)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

#%%

# PyTorch datasets
train_ds = ImageFolder(data_dir+'/train', train_tfms)
valid_ds = ImageFolder(data_dir+'/test', valid_tfms)

batch_size = 400

# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)


device = get_default_device()
device

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)

model = resnet.resnet18(pretrained=False,num_classes=10).to(device)
# model = resnet.resnet18(pretrained=False,num_classes=1000).to(device)



# %%
epochs = 8
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

checkpoint_path = 'resnet18_cifar10.pth'

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
    model.load_state_dict(checkpoint['model_state_dict'])
    model(next(iter(valid_dl))[0])
    resnet.collect_info_filt=False
    history = [evaluate(model, valid_dl)]
    print(history)


epochs = 8
max_lr = 0.5
grad_clip = 0.1
weight_decay = 0.
opt_func = torch.optim.Adam
l1_lambda = 1. #0.001
task_lambda = 1.
# checkpoint_path = 'resnet9_cifar10_trained.pth'
# checkpoint_path = 'resnet9_cifar10_trained_prune.pth'

# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'])

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
    model.epoch_end(epoch, result)
    history.append(result)
    writer.flush()

writer.close()
prune_model =resnet.resnet18(pretrained=False,num_classes=10).to(device)
prune_model.load_state_dict(model.state_dict())
prune_model = to_device(prune_model,device)
prune_model.eval()
clamp_th = 0.1

valid_filters =[]
for i in range(len(resnet.global_info_filt)):
    valid_filters.append(torch.sigmoid(resnet.global_info_filt[i]) >= clamp_th)

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