# -*- coding:utf-8 -*-
import torch
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

data_dir = r'E:\dev\facade-reading-data\facade-reading-data'
train_dir = os.path.join(data_dir, 'train')

train_split, validation_split, test_split = 0.6, 0.2, 0.2

batch_size = 8
data_loading_workers = 2
random_seed = 128

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
image_size = 224
image_resize = int(image_size * (256 / 224)), int(image_size * (256 / 224))

image_transform = transforms.Compose([
    transforms.Resize(image_resize),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

dataset = ImageFolder(root=train_dir, transform=image_transform)

train_size = int(len(dataset) * train_split)
validation_size = int(len(dataset) * validation_split)
test_size = len(dataset) - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size], generator=torch.Generator().manual_seed(random_seed))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)

if __name__ == '__main__':
    print(len(train_dataset), len(validation_dataset), len(test_dataset))