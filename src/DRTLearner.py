import math

import numpy as np
from torch import nn
import torch

from learner import Learner


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNetDRA(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrain=None):
        super(ResNetDRA, self).__init__()
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
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.finalLayer = torch.nn.Linear(2048, 5)  # 自己加上的

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.constant_(m.weight, 1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.new_param, self.init_param = [], []
        if pretrain:
            print('loading pretrain model from %s' % (pretrain))
            model = torch.load(pretrain)  # ['state_dict']
            prefix = 'module.features.'
            new_params = self.state_dict().copy()
            for x in new_params:
                if prefix + x in model:
                    new_params[x] = model[prefix + x]
                    self.init_param.append(x)
                else:
                    self.new_param.append(x)
                    print(x)
            self.load_state_dict(new_params)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, vars=None, bn_training=True):

        # 如果传入网络参数不为None，则重新设置网络参数，在forward完成之后得设置回来
        if vars != None:
            temp_para_list = nn.ParameterList(self.parameters())
            i = 0
            for params in self.parameters():
                params.data = vars[i].data.clone()
                i += 1


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # [9, 2048]

        # x = self.finalLayer(x)  # 不知道这一层需不需要在init中定义
        # print("hehehe:", x.shape)

        if vars != None:
            i = 0
            for params in self.parameters():
                params.data = temp_para_list[i].data.clone()
                i += 1

        return x


class BottleneckKDRA(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckKDRA, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        inp, oup = inplanes, planes * 4
        self.squeeze = inp // 16
        self.dim = int(math.sqrt(inp))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, self.squeeze, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.squeeze, 4, bias=False),
        )
        self.sf = nn.Softmax(dim=1)
        self.conv_s1 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s2 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s3 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)
        self.conv_s4 = nn.Conv2d(planes, planes, 1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
        y = self.sf(y)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        dyres = self.conv_s1(out) * y[:, 0] + self.conv_s2(out) * y[:, 1] + \
                self.conv_s3(out) * y[:, 2] + self.conv_s4(out) * y[:, 3]
        out = dyres + self.conv2(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # + dyres
        out = self.relu(out)

        return out


def resnet101_dy(pretrain=None):
    return ResNetDRA(BottleneckKDRA, [3, 4, 23, 3], pretrain=pretrain)





























