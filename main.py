import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import post_pruning

dataset_type ='cifar100' # 'LSVRC2012' # 'cifar10', 'cifar100'
model_name = 'resnet18' #'resnet18', 'resnet34', 'resnet50'

LSVRC2012_model_name_ckpt_map ={
                    'resnet18':  'resnet18_imagenet_mod.pth',
                   'resnet34':  'resnet34_imagenet_mod.pth',
                   'resnet50':  'resnet50_imagenet_mod.pth'
                   }
cifar10_model_name_ckpt_map ={
                   'resnet18':  'resnet18_cifar10.pth',
                   'resnet34':  None,
                   'resnet50':  None
                   }

cifar100_model_name_ckpt_map ={
                   'resnet18':  'resnet18_cifar100.pth',
                   'resnet34':  None,
                   'resnet50':  None
                   }

if dataset_type=='LSVRC2012':
    data_dir = '/media/Data/database/ImageNet_ILSVRC2012/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_tfms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    imagenet_val_db = torchvision.datasets.ImageNet(data_dir, split='val', transform=valid_tfms)
    # imagenet_val_db = torch.utils.data.Subset(imagenet_val_db, list(range(0, 5000)))
    train_size = int(0.6 * len(imagenet_val_db))
    val_size = len(imagenet_val_db) - train_size
    train_ds, valid_ds = torch.utils.data.random_split(imagenet_val_db, [train_size, val_size])
    batch_size = 16
    num_classes = 1000
    checkpoint_path = LSVRC2012_model_name_ckpt_map[model_name]
    summary_path = 'logs'
    train_net_before_pruning = False

elif dataset_type=='cifar10':
    data_dir = '/media/Data/database/cifar10'
    classes = os.listdir(data_dir + "/train")

    # Data transforms (normalization & data augmentation)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    #                          transforms.RandomHorizontalFlip(),
    #                          transforms.ToTensor(),
    #                          transforms.Normalize(*stats, inplace=True)])
    valid_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    # PyTorch datasets
    train_ds = ImageFolder(data_dir + '/train', valid_tfms)
    valid_ds = ImageFolder(data_dir + '/test', valid_tfms)
    batch_size = 400
    num_classes = 10
    checkpoint_path = cifar10_model_name_ckpt_map[model_name]
    summary_path = 'logs'
    train_net_before_pruning = False

elif dataset_type=='cifar100':

    data_dir = '/media/Data/database/cifar100'
    data_dir = '/media/ROIPO/Data/database/'
    # Data transforms (normalization & data augmentation)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(*stats, inplace=True)])
    valid_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tfms)
    valid_ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=valid_tfms)

    batch_size = 400
    num_classes = 100
    checkpoint_path = cifar100_model_name_ckpt_map[model_name]
    summary_path = 'logs'
    train_net_before_pruning = False
    fp16=False

else:
    raise Exception('Unknown dataset type!')



post_pruning.post_pruning(model_name=model_name,train_ds=train_ds,
                          valid_ds=valid_ds,dataset_type=dataset_type,num_classes=num_classes,
                          batch_size=batch_size,
                          checkpoint_path=checkpoint_path,
                          summary_path=summary_path,
                          train_net_before_pruning=train_net_before_pruning,
                          fp16=fp16)
