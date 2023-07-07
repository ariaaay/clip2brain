import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np

import sklearn.metrics


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # train_set = torchvision.datasets.CIFAR10(root="data",
    #                                          train=True,
    #                                          download=True,
    #                                          transform=train_transform)

    # test_set = torchvision.datasets.CIFAR10(root="data",
    #                                         train=False,
    #                                         download=True,
    #                                         transform=test_transform)

    train_set = torchvision.datasets.ImageNet(
        root="/lab_data/tarrlab/common/datasets/ILSVRC/Data/CLS-LOC",
        split="train",
        transform=train_transform,
    )

    test_set = torchvision.datasets.ImageNet(
        root="/lab_data/tarrlab/common/datasets/ILSVRC/Data/CLS-LOC",
        split="val",
        transform=test_transform,
    )

    # train_set = torchvision.datasets.place365(root="/lab_data/tarrlab/common/datasets/ILSVRC/Data/CLS-LOC",
    #                                          split="train",
    #                                          transform=train_transform)

    # test_set = torchvision.datasets.ImageNet(root="/lab_data/tarrlab/common/datasets/ILSVRC/Data/CLS-LOC",
    #                                         split="test",
    #                                         transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
    )

    classes = train_set.classes

    return train_loader, test_loader, classes
