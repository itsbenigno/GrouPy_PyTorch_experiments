import torch
import torch.nn as nn
import torch.nn.functional as F

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


class LeNet(nn.Module):

    def __str__(self):
        return "LeNet"

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 32, 5, padding=0)
        self.fc1 = nn.Linear(3200, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(x.size()[0], 3200)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class P4LeNet(nn.Module):

    def __str__(self):
        return "P4LeNet"

    def __init__(self):
        super(P4LeNet, self).__init__()
        self.conv1 = P4ConvZ2(1, 16, 5, padding=2)
        self.conv2 = P4ConvP4(16, 32, 5, padding=0)
        self.fc1 = nn.Linear(3200, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(plane_group_spatial_max_pooling(self.conv1(x), 2, 2))
        x = F.relu(plane_group_spatial_max_pooling(self.conv2(x), 2, 2))
        x = x.view(x.size()[0], 3200)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class P4MLeNet(nn.Module):

    def __str__(self):
        return "P4MLeNet"
    def __init__(self):
        super(P4MLeNet, self).__init__()
        self.conv1 = P4MConvZ2(1, 12, 5, padding=2)
        self.conv2 = P4MConvP4M(12, 16, 5, padding=0)
        self.fc1 = nn.Linear(3200, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(plane_group_spatial_max_pooling(self.conv1(x), 2, 2))
        x = F.relu(plane_group_spatial_max_pooling(self.conv2(x), 2, 2))
        x = x.view(x.size()[0], 3200)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)




def count_parameters(models):
    from prettytable import PrettyTable

    for model in models:
        print(str(model))
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")

count_parameters([LeNet(), P4LeNet(), P4MLeNet()])