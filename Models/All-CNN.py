import torch
import torch.nn as nn
import torch.nn.functional as F

#from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
#from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
#from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


class ALLCNNC(nn.Module):

    def __str__(self):
        return "ALLCNNC"

    def __init__(self):
        super(ALLCNNC, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1, stride=1, padding=0)
        self.conv9 = nn.Conv2d(192, 10, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        return F.log_softmax(x, dim=0)



model = ALLCNNC()
input = torch.rand(3,32,32)
print(model(input).shape)