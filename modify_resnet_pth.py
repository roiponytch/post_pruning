from collections import OrderedDict
import torch
import re

# checkpoint_path = 'resnet18_imagenet.pth'
# checkpoint_path = 'resnet34_imagenet.pth'
checkpoint_path = 'resnet50_imagenet.pth'

checkpoint = torch.load(checkpoint_path)
n_checkpoint = OrderedDict()

for k,v in checkpoint.items():
    nk = []
    ds_flag = False
    for s in k.split('.'):
        if 'conv' in s:
            nk.append(s)
            nk.append('conv_layer')
        elif 'bn' in s:
            nn = re.findall(r'\d+', s)[0]
            nk.append('conv'+str(nn))
            nk.append('bnorm')
        elif s.isnumeric():
            if ds_flag:
                if int(s)==0:
                    nk.append('conv_layer')
                else:
                    nk.append('bnorm')
            else:
                nk.append('layers')
                nk.append(str(s))
        else:
            if 'downsample'==s:
                ds_flag= True
            nk.append(s)

    n_checkpoint['.'.join(nk)] = v

new_ckpt_path =  checkpoint_path.split('.')
new_ckpt_path[0]=new_ckpt_path[0]+'_mod'
new_ckpt_path = '.'.join(new_ckpt_path)
torch.save(n_checkpoint ,new_ckpt_path)