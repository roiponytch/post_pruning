from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

# from .._internally_replaced_utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
# from ..utils import _log_api_usage_once
# from torchvision.utils import _log_api_usage_oncepi_usage_once
collect_info_filt = True
global_info_filt=[]

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        labels = labels.to(torch.int64)
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        labels = labels.to(torch.int64)
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


class conv_block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride = 1,padding=0, groups = 1, dilation = 1, activation =True,  pool=False, info_filt=None,):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        dilation=dilation)

        self.activation=activation
        self.pool = pool
        self.bnorm = nn.BatchNorm2d(out_planes)
        if self.activation:
            self.relu = nn.ReLU(inplace=True)

        if self.pool:
            self.pool_layer = nn.MaxPool2d(2)

        if info_filt ==None:
            self.info_filt = torch.nn.Parameter(
                torch.ones(size=[1,out_planes,1,1], requires_grad=True, device='cuda').mul(10.))
        else:
            self.info_filt = info_filt

    def forward(self, xb, use_info_filt = True):
            global collect_info_filt
            if use_info_filt:
                out = self.conv_layer(xb)*(torch.sigmoid(self.info_filt))
                # out = self.conv_layer(xb)* F.relu6(self.info_filt)
                # out = self.conv_layer(xb)*self.info_filt

            else:
                out = self.conv_layer(xb)

            if collect_info_filt:
                global_info_filt.append(self.info_filt)

            out = self.bnorm(out)
            if self.activation:
                out = self.relu(out)
            if self.pool:
                out = self.pool_layer(out)

            return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_block(inplanes,planes,kernel_size=3,stride=stride,padding=1)

        # self.conv2 = conv3x3(planes, planes)
        self.conv2 =conv_block(planes, planes,kernel_size=3,padding=1,activation=False)
        self.downsample = downsample
        #
        if downsample:
            del self.downsample.info_filt
            self.downsample.info_filt = self.conv2.info_filt

        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor,use_info_filt: bool=True) -> Tensor:
        identity = x

        out = self.conv1(x,use_info_filt =use_info_filt)
        out = self.conv2(out,use_info_filt =use_info_filt)

        if self.downsample is not None:
            identity = self.downsample(x,use_info_filt =use_info_filt)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_block(inplanes, width, kernel_size=1)
        # self.conv1 = conv1x1(inplanes, width)
        # self.bn1 = norm_layer(width)

        self.conv2 = conv_block(width, width,kernel_size=3, stride=stride,groups= groups,dilation= dilation,padding=dilation)
        # self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)

        self.conv3 = conv_block(width, planes * self.expansion,kernel_size=1,activation=False)
        # self.conv3 = conv1x1(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        #
        if downsample:
            del self.downsample.info_filt
            self.downsample.info_filt = self.conv3.info_filt


        self.relu = nn.ReLU(inplace=True)
        self.stride = stride


    def forward(self, x: Tensor,use_info_filt: bool=True) -> Tensor:
        identity = x

        out = self.conv1(x,use_info_filt =use_info_filt)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.conv2(out,use_info_filt =use_info_filt)
        # out = self.bn2(out)
        # out = self.relu(out)

        out = self.conv3(out,use_info_filt =use_info_filt)
        # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x,use_info_filt =use_info_filt)

        out += identity
        out = self.relu(out)

        return out


class LayerModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x,use_info_filt=True):
        for layer in self.layers:
            x = layer(x, use_info_filt=use_info_filt)
        return x

class ResNet(ImageClassificationBase):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dataset_type: str = 'LSVRC2012'
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.use_info_filt= True
        if dataset_type =='LSVRC2012':
            self.conv1 = conv_block(3, self.inplanes, kernel_size=7, stride=2, padding=3)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif dataset_type =='cifar10' or dataset_type =='cifar100':
            self.conv1 = conv_block(3, self.inplanes, kernel_size=3, stride=1, padding=1)
            self.maxpool = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = LayerModule(self._make_layer(block, 64, layers[0]))
        self.layer2 = LayerModule(self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]))
        self.layer3 = LayerModule(self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]))
        self.layer4 = LayerModule(self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


        self.info_filt_params=[]
        self.model_params = []

        for name, param in self.named_parameters():
            if name.endswith('info_filt'):
                self.info_filt_params.append(param)
            else:
                self.model_params.append(param)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv_block(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, activation=False)
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        # return nn.Sequential(*layers)
        return layers

    def _forward_impl(self, x: Tensor,use_info_filt =None) -> Tensor:
        # See note [TorchScript super()]
        if use_info_filt == None:
            use_info_filt = self.use_info_filt
        x = self.conv1(x,use_info_filt =use_info_filt)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)

        # for l in self.layer1:
        #     x=l(x,use_info_filt =use_info_filt)
        # for l in self.layer2:
        #     x=l(x,use_info_filt =use_info_filt)
        # for l in self.layer3:
        #     x = l(x, use_info_filt=use_info_filt)
        # for l in self.layer4:
        #     x=l(x,use_info_filt =use_info_filt)
        x = self.layer1(x,use_info_filt =use_info_filt)
        x = self.layer2(x,use_info_filt =use_info_filt)
        x = self.layer3(x,use_info_filt =use_info_filt)
        x = self.layer4(x,use_info_filt =use_info_filt)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor,use_info_filt = None) -> Tensor:
        return self._forward_impl(x, use_info_filt=use_info_filt)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)