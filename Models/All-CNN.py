import torch
import torch.nn as nn
import torch.nn.functional as F

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M


class AllCNNC(nn.Module):

    def __str__(self):
        return "ALLCNNC"

    def __init__(self):
        super(AllCNNC, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 192, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv6_bn = nn.BatchNorm2d(192)
        self.conv7 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, 1, stride=1, padding=0)
        self.conv8_bn = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, 10, 1, stride=1, padding=0)
        self.conv9_bn = nn.BatchNorm2d(10)

    def forward(self, x):
        x = F.dropout(x, p=0.2, training=True)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.relu(self.conv9_bn(self.conv9(x)))

        x = torch.sum(x, dim=-1)
        x = torch.sum(x, dim=-1)
        x /= 8*8


        return F.log_softmax(x, dim=1) #is it right?



class P4AllCNNC(nn.Module):

    def __str__(self):
        return "P4ALLCNNC"

    def __init__(self):
        super(P4AllCNNC, self).__init__()
        self.conv1 = P4ConvZ2(3, 48, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(48*4)
        self.conv2 = P4ConvP4(48, 48, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(48*4)
        self.conv3 = P4ConvP4(48, 48, 3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(48*4)
        self.conv4 = P4ConvP4(48, 96, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(96*4)
        self.conv5 = P4ConvP4(96, 96, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(96*4)
        self.conv6 = P4ConvP4(96, 96, 3, stride=2, padding=1)
        self.conv6_bn = nn.BatchNorm2d(96*4)
        self.conv7 = P4ConvP4(96, 96, 3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(96*4)
        self.conv8 = P4ConvP4(96, 96, 1, stride=1, padding=0)
        self.conv8_bn = nn.BatchNorm2d(96*4)
        self.conv9 = P4ConvP4(96, 10, 1, stride=1, padding=0)
        self.conv9_bn = nn.BatchNorm2d(10*4)

    def forward(self, x):
        x = F.dropout(x, p=0.2, training=True)
        x = self.conv1(x)
        print("shape after first convolution",x.shape)
        x = x.view(x.shape[0], x.shape[1]*4, 32, 32)
        x = F.relu(self.conv1_bn(x))
        x = x.view(x.shape[0], 4, -1, 32, 32)

        """
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.relu(self.conv9_bn(self.conv9(x)))
        """
        x = torch.sum(x, dim=-1)
        x = torch.sum(x, dim=-1)
        x = torch.sum(x, dim=-1)
        x /= 8*8 *4


        return F.log_softmax(x, dim=1) #is it right?


class P4MAllCNNC(nn.Module):

    def __str__(self):
        return "P4MALLCNNC"

    def __init__(self):
        super(P4MAllCNNC, self).__init__()
        self.conv1 = P4MConvZ2(3, 32, 3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = P4MConvP4M(32, 32, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = P4MConvP4M(32, 32, 3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = P4MConvP4M(32, 64, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = P4MConvP4M(64, 64, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = P4MConvP4M(64, 64, 3, stride=2, padding=1)
        self.conv6_bn = nn.BatchNorm2d(64)
        self.conv7 = P4MConvP4M(64, 64, 3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8 = P4MConvP4M(64, 64, 1, stride=1, padding=0)
        self.conv8_bn = nn.BatchNorm2d(64)
        self.conv9 = P4MConvP4M(64, 10, 1, stride=1, padding=0)
        self.conv9_bn = nn.BatchNorm2d(10)

    def forward(self, x):
        x = F.dropout(x, p=0.2, training=True)
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.relu(self.conv9_bn(self.conv9(x)))

        x = torch.sum(x, dim=-1)
        x = torch.sum(x, dim=-1)
        x = torch.sum(x, dim=-1)
        x /= 8*8 *8


        return F.log_softmax(x, dim=1) #is it right?

#model1 = AllCNNC()
model2 = P4AllCNNC()
input = torch.rand(10,3,32,32)
#print("Model 1\n",model1(input).shape)
print("Model 2 shape",model2(input).shape)