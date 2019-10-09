from __future__ import absolute_import

import torch
import math
import copy
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F

from models import inflate
from models import non_local

__all__ = ['VidNonLocalResNet50', 'ImgResNet50', 'Classifier']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Bottleneck3d(nn.Module):

    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VidNonLocalResNet50(nn.Module):

    def __init__(self, **kwargs):
        super(VidNonLocalResNet50, self).__init__()

        resnet2d = torchvision.models.resnet50(pretrained=True)
        resnet2d.layer4[0].conv2.stride=(1,1)
        resnet2d.layer4[0].downsample[0].stride=(1,1) 

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        
        self.layer1 = self._inflate_reslayer(resnet2d.layer1)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, nonlocal_idx=[1,3], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, nonlocal_idx=[1,3,5], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4)

    def _inflate_reslayer(self, reslayer2d, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i,layer2d in enumerate(reslayer2d):
            layer3d = Bottleneck3d(layer2d)
            reslayers3d.append(layer3d)

            if i in nonlocal_idx:
                non_local_block = non_local.NONLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        f = F.avg_pool2d(x, x.size()[2:])
        f = f.view(b, t, -1)
        frame_feature = f.view(b*t, -1)

        if not self.training:
            return f

        f = f.mean(1)

        return f, frame_feature


class ImgResNet50(nn.Module):
    def __init__(self, **kwargs):
        super(ImgResNet50, self).__init__()

        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride=(1,1)
        resnet.layer4[0].downsample[0].stride=(1,1) 
        self.base = nn.Sequential(*list(resnet.children())[:-2])
         

    def forward(self, x):
        x = self.base(x)
        f = F.avg_pool2d(x, x.size()[2:])
        f = f.view(f.size(0), -1)

        return f


class Classifier(nn.Module):
    def __init__(self, num_classes=625):
        super(Classifier, self).__init__()

        # classifier using Random initialization
        self.classifier = nn.Linear(2048, num_classes)
        self.classifier.apply(weights_init_classifier)

    def forward(self, f):
        y = self.classifier(f)

        return y
