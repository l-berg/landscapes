import torch
from torch import nn
from torchvision import datasets, models, transforms
import numpy as np
import random

import time
import pickle
import os
import argparse
from pathlib import Path


def set_seed(seed: int, device=None):
    torch.manual_seed(seed)
    if device is not None and device == 'cuda:0':
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data

def save_str(path: str, data: str):
    with open(path, 'w') as f:
        f.write(data)

def get_model(name, in_channels=3, in_width=32, num_classes=10):
    if name == 'resnet':
        model = models.ResNet(models.resnet.BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif name == 'vgg':
        model = models.vgg11(num_classes=num_classes)
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if in_width < 32:
            # remove last max-pooling layer. now mnist (28x28) works (as long as >=16x16)
            model.features = nn.Sequential(*(list(model.features)[:-1]))
        # import ipdb; ipdb.set_trace()
    else:
        raise IndexError(f'Unknown model type: {name}')
    return model


def get_dataloaders(name, shuffle=True, pin_memory=True, batch_size=256, nworkers=4):
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    imagenet_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if name == 'fashion-mnist':
        print('Fetching FashionMNIST')
        trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=mnist_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=pin_memory,
                                                   shuffle=shuffle, num_workers=nworkers)
        valset = datasets.FashionMNIST('./data', train=False, transform=mnist_transforms)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, pin_memory=pin_memory,
                                                 shuffle=shuffle, num_workers=nworkers)
    elif name == 'mnist':
        print('Fetching MNIST')
        trainset = datasets.MNIST('./data', train=True, download=True, transform=mnist_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=pin_memory,
                                                   shuffle=shuffle, num_workers=nworkers)
        valset = datasets.MNIST('./data', train=False, transform=mnist_transforms)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, pin_memory=pin_memory,
                                                 shuffle=shuffle, num_workers=nworkers)
    elif name == 'cifar-10':
        print('Fetching CIFAR10')
        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=imagenet_transforms)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, pin_memory=pin_memory,
                                                   shuffle=shuffle, num_workers=nworkers)
        valset = datasets.CIFAR10('./data', train=False, transform=imagenet_transforms)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, pin_memory=pin_memory,
                                                 shuffle=shuffle, num_workers=nworkers)
    else:
        raise IndexError(f'Unknown dataset: {name}')
    return train_loader, val_loader
