'''
Author : Na Rae Baek
Author email : naris27@dgu.ac.kr
Github : https://github.com/ban2aru
License : MIT license
Modified Year-Month : 2022-04

Description : ResNet.py
The main code for ResNet model using Pytorch

Reference :
    Deep Residual Learning for Image Recognition
    HE, Kaiming, et al. Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. p. 770-778.
    https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
'''

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']


model_urls = {
    'ResNet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth',
    'ResNet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'ResNet50': 'http://download.pytorch.org/models/resnet50-19c8e357.pth',
    'ResNet101': 'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ResNet152': 'http://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def Shortcut_layer(in_channels, out_channels, stride, expansion) :
    model = nn.Sequential()

    #if input dimension and output dimension of shortcut are different, use 1x1 conv for matching dimension 
    if stride != 1 or in_channels != out_channels*expansion :
        model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels*expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels*expansion)
        )

    return model


class ResidualBlock(nn.Module):
# use this block for ResNet-18, 34
    expansion=1

    def __init__(self, in_channels, out_channels, stride=1, padding_mode='zeros'):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, dilation=1, padding_mode=padding_mode, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*ResidualBlock.expansion, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode=padding_mode, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*ResidualBlock.expansion)

        self.downsample = Shortcut_layer(in_channels=in_channels, out_channels=out_channels, stride=stride, expansion=ResidualBlock.expansion)
       
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class BottleNeckBlock(nn.Module):
# use this block for ResNet-50 over
    expansion=4

    def __init__(self, in_channels, out_channels, stride=1, padding_mode='zeros'):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*BottleNeckBlock.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*BottleNeckBlock.expansion)

        self.downsample = Shortcut_layer(in_channels=in_channels, out_channels=out_channels, stride=stride, expansion=BottleNeckBlock.expansion)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block_type, num_blocks, num_classes=1000) :
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layers(block_type, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block_type, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block_type, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block_type, 512, num_blocks[3], stride=2)

        self.fc = nn.Linear(512 * block_type.expansion, num_classes)

    def _make_layers(self, block_type, out_channels, num_blocks, stride):
        #First layer of blocks has 1 or 2 strides. (The others are 1)
        stride_list = [stride] + [1] * (num_blocks-1)
        Conv_blocks = []
        for stride in stride_list:
            Conv_blocks.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block_type.expansion
        return nn.Sequential(*Conv_blocks)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def _resnet(depth) :
    depth_list = [18, 34, 50, 101, 152]
    assert (depth in depth_list), "ResNet's depth has only [18, 34, 50, 101, 152]"
    resnet_dict = {
        '18' : (ResidualBlock, [2,2,2,2]),
        '34' : (ResidualBlock, [3,4,6,3]),
        '50' : (BottleNeckBlock, [3,4,6,3]),
        '101' : (BottleNeckBlock, [3,4,23,3]),
        '152' : (BottleNeckBlock, [3,8,36,3]),
    }

    return resnet_dict[str(depth)]

def ResNet18(pretrained_model=False, num_classes=1000, **kwargs):
    block_type, num_blocks = _resnet(depth=18)
    model = ResNet(block_type=block_type, num_blocks=num_blocks, num_classes=num_classes, **kwargs)
    if pretrained_model :
        print("| Downloading ResNet-18 Model with ImageNet |")
        imagenet_model = model_zoo.load_url(model_urls['ResNet18'])
        imagenet_model.pop('fc.weight', None)
        imagenet_model.pop('fc.bias', None)
        model.load_state_dict(imagenet_model, strict=False)

    return model

def ResNet34(pretrained_model=False, num_classes=1000, **kwargs):
    block_type, num_blocks = _resnet(depth=34)
    model = ResNet(block_type=block_type, num_blocks=num_blocks, num_classes=num_classes, **kwargs)
    if pretrained_model :
        print("| Downloading ResNet-34 Model with ImageNet |")
        imagenet_model = model_zoo.load_url(model_urls['ResNet34'])
        imagenet_model.pop('fc.weight', None)
        imagenet_model.pop('fc.bias', None)
        model.load_state_dict(imagenet_model, strict=False)

    return model

def ResNet50(pretrained_model=False, num_classes=1000, **kwargs):
    block_type, num_blocks = _resnet(depth=50)
    model = ResNet(block_type=block_type, num_blocks=num_blocks, num_classes=num_classes, **kwargs)
    if pretrained_model :
        print("| Downloading ResNet-50 Model with ImageNet |")
        imagenet_model = model_zoo.load_url(model_urls['ResNet50'])
        imagenet_model.pop('fc.weight', None)
        imagenet_model.pop('fc.bias', None)
        model.load_state_dict(imagenet_model, strict=False)

    return model

def ResNet101(pretrained_model=False, num_classes=1000, **kwargs):
    block_type, num_blocks = _resnet(depth=101)
    model = ResNet(block_type=block_type, num_blocks=num_blocks, num_classes=num_classes, **kwargs)
    if pretrained_model :
        print("| Downloading ResNet-101 Model with ImageNet |")
        imagenet_model = model_zoo.load_url(model_urls['ResNet101'])
        imagenet_model.pop('fc.weight', None)
        imagenet_model.pop('fc.bias', None)
        model.load_state_dict(imagenet_model, strict=False)

    return model

def ResNet152(pretrained_model=False, num_classes=1000, **kwargs):
    block_type, num_blocks = _resnet(depth=152)
    model = ResNet(block_type=block_type, num_blocks=num_blocks, num_classes=num_classes, **kwargs)
    if pretrained_model :
        print("| Downloading ResNet-152 Model with ImageNet |")
        imagenet_model = model_zoo.load_url(model_urls['ResNet152'])
        imagenet_model.pop('fc.weight', None)
        imagenet_model.pop('fc.bias', None)
        model.load_state_dict(imagenet_model, strict=False)

    return model


