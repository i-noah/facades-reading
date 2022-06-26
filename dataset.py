# -*- coding:utf-8 -*-
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from config import (image_size, train_dir, train_split, batch_size, 
                    random_seed, validation_split, data_loading_workers)
image_resize = int(image_size * (256 / 224)), int(image_size * (256 / 224))

image_transform = T.Compose([
    T.Resize(image_resize),
    T.CenterCrop(image_size),
    T.ToTensor(),
    # 使用RGB通道的均值和标准差，以标准化每个通道
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=train_dir, transform=image_transform)
classes_cnt = len(dataset.classes)

train_size = int(len(dataset) * train_split)
validation_size = int(len(dataset) * validation_split)
test_size = len(dataset) - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size], 
                    generator=torch.Generator().manual_seed(random_seed))

train_loader = DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)

validation_loader = DataLoader(
    validation_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)

if __name__ == '__main__':
    print(train_dataset.indices)
    print(validation_dataset.indices)
    print(test_dataset.indices)
    print(len(train_dataset), len(validation_dataset), len(test_dataset))