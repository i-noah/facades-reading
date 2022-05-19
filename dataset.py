# -*- coding:utf-8 -*-
import torch
import os
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def get_train_transform(mean=mean, std=std, size=0):
    train_transform = transforms.Compose([
        transforms.Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform

def get_test_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        transforms.Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

data_dir = r'E:\dev\facade-reading-data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
valid_size = 0.2
batch_size = 8
data_loading_workers = 2
random_seed = 128

# load the dataset
train_dataset = datasets.ImageFolder(
    root=train_dir, transform=get_train_transform(size=300),
)

valid_dataset = datasets.ImageFolder(
    root=train_dir, transform=get_test_transform(size=300),
)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler,
    num_workers=data_loading_workers, pin_memory=True,
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, sampler=valid_sampler,
    num_workers=data_loading_workers, pin_memory=True,
)

test_dataset = datasets.ImageFolder(
    root=test_dir, transform=get_test_transform(size=300),
)

if __name__ == '__main__':
    print(train_dataset.class_to_idx)