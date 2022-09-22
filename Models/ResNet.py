import torch
import torch.nn as nn
import torch.nn.functional as F

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_avg_pooling

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential() #identity
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        print(out.shape)
        out = self.layer1(out)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = F.avg_pool2d(out, out.size()[3])
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.linear(out)
        return out


def ResNet44():
    return ResNet(BasicBlock, [7, 7, 7])





class P4BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(P4BasicBlock, self).__init__()
        self.conv1 = P4ConvP4(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = P4ConvP4(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential() #identity
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                P4ConvP4(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class P4ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(P4ResNet, self).__init__()
        self.in_planes = 8

        self.conv1 = P4ConvZ2(3, 8, kernel_size=3, stride=1, padding=1, bias=False) #first convolution must be from Z2
        self.bn1 = nn.BatchNorm3d(8)
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        print(out.shape)
        out = self.layer1(out)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = plane_group_spatial_avg_pooling(out, out.size()[4]) #in order to reduce filter to 1x1
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.linear(out)
        return out


def P4ResNet44():
    return P4ResNet(P4BasicBlock, [7, 7, 7])





class P4MBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(P4MBasicBlock, self).__init__()
        self.conv1 = P4MConvP4M(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = P4MConvP4M(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential() #identity
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                P4MConvP4M(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class P4MResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(P4MResNet, self).__init__()
        self.in_planes = 6

        self.conv1 = P4MConvZ2(3, 6, kernel_size=3, stride=1, padding=1, bias=False) #first convolution must be from Z2
        self.bn1 = nn.BatchNorm3d(6)
        self.layer1 = self._make_layer(block, 6, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 11, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 23, num_blocks[2], stride=2)
        self.linear = nn.Linear(184, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        print(out.shape)
        out = self.layer1(out)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = plane_group_spatial_avg_pooling(out, out.size()[4]) #in order to reduce filter to 1x1
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.linear(out)
        return out


def P4MResNet44():
    return P4MResNet(P4MBasicBlock, [7, 7, 7])


#model = ResNet44()
#model = P4ResNet44()
#model = P4MResNet44()
#input = torch.rand(10,3,224,224)
#print(model(input).shape)