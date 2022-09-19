import numpy as np

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

from Models.LeNet import *

#reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)

torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
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


def get_model(model):
  actual_model = model()
  return actual_model, optim.SGD(actual_model.parameters(), lr=lr, momentum=momentum)
#may consider to use Adam


def step(model, loss_func, input, target, opt=None):
  loss = loss_func(model(input), target)

  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()

  return loss.item(), len(input)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for input, target in train_dl:
            step(model, loss_func, input, target, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[step(model, loss_func, input, target) for input, target in valid_dl])
            #mean error
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            print(epoch, val_loss)

def test(model, test_dl):
    correct = 0
    total = 0
    correct_pred = {number: 0 for number in range(10)}
    total_pred = {number: 0 for number in range(10)}
    with torch.no_grad():
        for input, targets in test_dl:
            output = model(input)
            predictions = torch.max(output, dim=1)[1]
            correct += (predictions == targets).sum().item()
            total += len(input)
            for target, prediction in zip(targets, predictions):
                if target == prediction:
                    correct_pred[target.item()] += 1
                total_pred[target.item()] += 1

    """for number in correct_pred.keys():
        print("Class: ", number, "Correct predictions: ", correct_pred[number], "/", total_pred[number])

    print(correct, "/", total)"""

    return correct, total, correct_pred, total_pred

import json
import os

#create (if it doesn't already exists) test folder
try:
    os.mkdir("test_results/MINST")
except OSError as error:
    print(error)

models = [P4MLeNet]
for model_to_test in models:
    temp_model_instance = model_to_test()

    total_params = 0
    for name, parameter in temp_model_instance.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params

    model_name = str(temp_model_instance)

    del temp_model_instance

    loss_func = F.nll_loss
    model, opt = get_model(model_to_test)
    #fit(epochs, model, loss_func, opt, train_loader, test_loader)
    correct_pred, total_pred, class_correct_pred, object_per_class = test(model, test_loader)

    results = {}
    results["accuracy_pct"] = correct_pred/total_pred * 100.0
    results["correct_pred"] = correct_pred
    results["total_pred"] = total_pred
    results["parameters"] = total_params
    results["class_correct_pred"] = class_correct_pred
    results["object_per_class"] = object_per_class

    # Serializing json
    json_object = json.dumps(results)

    # save #parameters, accuracy on fixed seed, pytorch model
    with open("test_results/MINST/"+model_name+"-results.json", 'w') as outfile:
        outfile.write(json_object)

    torch.save(model.state_dict(), "test_results/MINST/"+model_name)