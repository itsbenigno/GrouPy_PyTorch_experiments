import torch
import torch.nn as nn
import torch.nn.functional as F

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


class Z2CNN(nn.Module):
    def __str__(self):
        return "Z2CNN"

    def __init__(self):
        super(Z2CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(20)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv4_bn = nn.BatchNorm2d(20)
        self.conv5 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv5_bn = nn.BatchNorm2d(20)
        self.conv6 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv6_bn = nn.BatchNorm2d(20)
        self.conv7 = nn.Conv2d(20, 10, kernel_size=4, padding=1)
        self.pool = nn.MaxPool2d(2,2)


    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = self.conv7(x)
        x = torch.sum(x, dim=-1)
        x = torch.sum(x, dim=-1)

        return F.log_softmax(x, dim=1)



class P4CNN(nn.Module):
    def __str__(self):
        return "P4CNN"

    def __init__(self):
        super(P4CNN, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv1_bn = nn.BatchNorm3d(10)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv2_bn = nn.BatchNorm3d(10)
        self.conv3 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3_bn = nn.BatchNorm3d(10)
        self.conv4 = P4ConvP4(10, 10, kernel_size=3)
        self.conv4_bn = nn.BatchNorm3d(10)
        self.conv5 = P4ConvP4(10, 10, kernel_size=3)
        self.conv5_bn = nn.BatchNorm3d(10)
        self.conv6 = P4ConvP4(10, 10, kernel_size=3)
        self.conv6_bn = nn.BatchNorm3d(10)
        self.conv7 = P4ConvP4(10, 10, kernel_size=4, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = self.conv7(x)
        x = torch.sum(x, dim=-3)
        x = torch.sum(x, dim=-1)
        x = torch.sum(x, dim=-1)

        return F.log_softmax(x, dim=1)




class earlyP4CNN(nn.Module):
    def __str__(self):
        return "earlyP4CNN"

    def __init__(self):
        super(earlyP4CNN, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv1_bn = nn.BatchNorm3d(10)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv2_bn = nn.BatchNorm3d(10)
        self.conv3 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3_bn = nn.BatchNorm3d(10)
        self.conv4 = P4ConvP4(10, 10, kernel_size=3)
        self.conv4_bn = nn.BatchNorm3d(10)
        self.conv5 = nn.Conv2d(40, 20, kernel_size=3)
        self.conv5_bn = nn.BatchNorm2d(20)
        self.conv6 = nn.Conv2d(20, 20, kernel_size=3)
        self.conv6_bn = nn.BatchNorm2d(20)
        self.conv7 = nn.Conv2d(20, 10, kernel_size=4, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = x.view([x.shape[0], -1, 8, 8])
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = self.conv7(x)
        x = torch.sum(x, dim=-1)
        x = torch.sum(x, dim=-1)

        return F.log_softmax(x, dim=1)






class EP4CNN(nn.Module):
    def __str__(self):
        return "EP4CNN"

    def __init__(self):
        super(EP4CNN, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv1_bn = nn.BatchNorm3d(10)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv2_bn = nn.BatchNorm3d(10)
        self.conv3 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3_bn = nn.BatchNorm3d(10)
        self.conv4 = P4ConvP4(10, 10, kernel_size=3)
        self.conv4_bn = nn.BatchNorm3d(10)
        self.conv5 = P4ConvP4(10, 10, kernel_size=3)
        self.conv5_bn = nn.BatchNorm3d(10)
        self.conv6 = P4ConvP4(10, 10, kernel_size=3)
        self.conv6_bn = nn.BatchNorm3d(10)
        self.conv7 = P4ConvP4(10, 10, kernel_size=4, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = self.conv7(x)
        x = torch.sum(x, dim=-3)
        x = torch.sum(x, dim=-1)
        x = torch.sum(x, dim=-1)

        return F.log_softmax(x, dim=1)
