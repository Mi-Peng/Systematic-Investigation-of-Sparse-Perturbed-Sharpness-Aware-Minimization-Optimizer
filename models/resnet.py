'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.configurable import configurable

from models.build import MODELS_REGISTRY

from models.nm_conv import SAMConv

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_type, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_type(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_type(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_type, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_type(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_type(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, conv_type=nn.Conv2d):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv_type(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], conv_type, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], conv_type, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], conv_type, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], conv_type, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, conv_type, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, conv_type, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,y=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if y is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(out,y.view(-1))
            return loss
        else:
            return out


def _cfg_to_resnet(args):
    return {
        "num_classes": args.n_classes,
        "imagenet": args.dataset[:8] == 'ImageNet',
        "samconv": args.samconv, 
    }

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet18(num_classes=10, imagenet=False, samconv=False):
    if imagenet: 
        return torchvision.models.resnet18(pretrained=False)
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes, conv_type=SAMConv if samconv else nn.Conv2d)

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet34(num_classes=10, imagenet=False, samconv=False):
    if imagenet: 
        return torchvision.models.resnet34(pretrained=False)
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes, conv_type=SAMConv if samconv else nn.Conv2d)

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet50(num_classes=10, imagenet=False, samconv=False):
    if imagenet: 
        model = torchvision.models.resnet50(pretrained=False)
        if samconv:
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    module.__class__ = SAMConv
        return model
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes = num_classes, conv_type=SAMConv if samconv else nn.Conv2d)

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet101(num_classes=10, imagenet=False, samconv=False):
    if imagenet: return torchvision.models.resnet101(pretrained=False)
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes = num_classes, conv_type=SAMConv if samconv else nn.Conv2d)

@MODELS_REGISTRY.register()
@configurable(from_config=_cfg_to_resnet)
def resnet152(num_classes=10, imagenet=False, samconv=False):
    if imagenet: return torchvision.models.resnet152(pretrained=False)
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes = num_classes, conv_type=SAMConv if samconv else nn.Conv2d)


def test():
    net = resnet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
model = torchvision.models.resnet50()