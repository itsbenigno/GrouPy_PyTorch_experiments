import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

#reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)

torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True  # Note that this Deterministic mode can have a performance impact
torch.backends.cudnn.benchmark = False

#CONFIGURATION
batch_size = 64 #int
test_batch_size = 1000 #int
epochs = 10 #int
lr = 0.01 #float
momentum = 0.5 #float
cuda = False #boolean
log_interval = 10 #int

#dataset transformation
dataset_mean = 0.1307
dataset_std = 0.3081
image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean = dataset_mean, std = dataset_std)
    ]
)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=image_transforms), batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=image_transforms), batch_size=test_batch_size, shuffle=True, **kwargs)


class MNIST_NET(nn.Module):
    def __init__(self):
        super(MNIST_NET, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3)
        self.conv2 = P4ConvP4(10, 10, kernel_size=3)
        self.conv3 = P4ConvP4(10, 20, kernel_size=3)
        self.conv4 = P4ConvP4(20, 20, kernel_size=3)
        self.fc1 = nn.Linear(4 * 4 * 20 * 4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def get_model():
  model = MNIST_NET()
  return model, optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
  for epoch in range(epochs):
    model.train()
    for input, target in train_dl:
        step(model, loss_func, input, target, opt)

    model.eval()
    with torch.no_grad():
      losses, nums = zip(
        *[step(model, loss_func, input, target) for input, target in valid_dl]
      )
    val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

    print(epoch, val_loss)


def step(model, loss_func, input, target, opt=None):
  loss = F.nll_loss(model(input), target)

  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()

  return loss.item(), len(input)

loss_func = F.nll_loss
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_loader, test_loader)