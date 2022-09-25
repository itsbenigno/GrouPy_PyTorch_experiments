#importing dependencies
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

import json
import os

import utility
import utility as counting
import Models.LeNet as Models

#reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)


def train_data_mean_std():
    dataset = datasets.MNIST(root='../data', train=True, download=True)
    data = dataset.data / 255
    mean = data.mean() 
    std = data.std()
    return mean, std

#retrieve the datasets
def get_datasets(train_bs, val_bs, test_bs):

    # dataset transformation
    dataset_mean, dataset_std = train_data_mean_std()

    image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ]
    )

    train_dataset = datasets.MNIST(root='../data', train=True, transform=image_transforms, download=True)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

    test_dataset = datasets.MNIST(root='./data', train=False, transform=image_transforms, download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_bs)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=val_bs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs)

    return train_loader, validation_loader, test_loader


#retrieving the model and initializing the optimizer
def get_model(model, lr, momentum=None):
  actual_model = model()
  return actual_model, optim.SGD(actual_model.parameters(), lr=lr, momentum=momentum)


#compute loss given loss function, model and input
def step(model, loss_func, input, target, opt=None):
  loss = loss_func(model(input), target)

  if opt is not None:
    loss.backward()
    opt.step()
    opt.zero_grad()

  return loss.item()


#training of the model
def fit(epochs, model, loss_func, opt, train_dl, valid_dl, classes, max_patience):
    path = "test_results/MINST/checkpoint-"+str(model)
    best_val_loss = 100.
    patience = max_patience

    for epoch in range(epochs):
        print("Epoch ", epoch)
        model.train()
        for input, target in train_dl:
            loss = step(model, loss_func, input, target, opt)
            
        print("Train loss: ",loss)
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for input, targets in valid_dl:
                val_loss = loss_func(model(input), targets).item()

        print("Validation loss: ", val_loss)

        
        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            }, path)
            patience = max_patience
        else:
            patience -= 1
            if (patience == 0):
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint['model_state_dict'])
                opt.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Stopping training and taking the model that performed best in the validation set")
                return None

    #pick the model with the best val_loss
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])



def test(model, test_dl, classes):
    correct = 0
    total = 0
    correct_pred = {pred_class: 0 for pred_class in classes}
    total_pred = {pred_class: 0 for pred_class in classes}
    model.eval()
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

    return correct, total, correct_pred, total_pred


#create (if it doesn't already exists) test folder
def create_test_dir(directory_path):
    try:
        os.mkdir("test_results/"+directory_path)
    except OSError as error:
        print(error)


def testing():
    # CONFIGURATION
    train_batch_size = 128  
    val_batch_size = 10000
    test_batch_size = 1000 
    epochs = 25  
    lr = 0.05
    momentum = 0.9 
    classes = [0,1,2,3,4,5,6,7,8,9]
    max_patience = 15

    create_test_dir("MINST")
    train_loader, val_loader, test_loader = get_datasets(train_batch_size, val_batch_size, test_batch_size)
    models = [Models.LeNet, Models.P4LeNet, Models.P4MLeNet]

    for model_to_test in models:

        temp_model_instance = model_to_test()

        model_params = utility.count_parameters(temp_model_instance)
        model_name = str(temp_model_instance)
        print(model_name)
        del temp_model_instance

        loss_func = F.nll_loss
        model, opt = get_model(model_to_test, lr, momentum)
        fit(epochs, model, loss_func, opt, train_loader, val_loader, classes, max_patience)
        correct_pred, total_pred, class_correct_pred, object_per_class = test(model, test_loader, classes)

        results = {}
        results["accuracy_pct"] = (correct_pred/total_pred) * 100.0
        results["correct_pred"] = correct_pred
        results["total_pred"] = total_pred
        results["parameters"] = model_params
        results["class_correct_pred"] = class_correct_pred
        results["object_per_class"] = object_per_class

        # Serializing json
        json_object = json.dumps(results)
        # save #parameters, accuracy on fixed seed, pytorch model
        with open("test_results/MINST/"+model_name+"-results.json", 'w') as outfile:
            outfile.write(json_object)

        torch.save(model.state_dict(), "test_results/MINST/"+model_name)

testing()