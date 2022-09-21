from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M

import torch

input = torch.rand(2,1,3,3)

conv1 = P4ConvZ2(1, 2, 2)

print(conv1(input).shape)