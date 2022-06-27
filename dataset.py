# -*- coding:utf-8 -*-
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from config import (image_size, train_dir, train_split, batch_size, 
                    random_seed, validation_split, data_loading_workers)

# 图像预处理列表
image_transform = T.Compose([
    T.Resize(256), # 缩放到256像素宽，高度按比例
    T.CenterCrop(image_size), # 按中心裁剪大小为image_size的图片
    T.ToTensor(), # 转换为tensor，映射到区间，[0, 1]
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 使用RGB通道的均值和标准差，以标准化每个通道
])

"""
根据图片文件夹加载数据集
角度正/
├─ 0/
│  ├─ ... # n 张拍摄角度不正的图片
│  └─ *.jpg
└─ 1/
    ├─ ... # n 张拍摄角度正的图片
    └─ *.jpg
"""
dataset = ImageFolder(root=train_dir, transform=image_transform)

# 标签/分类数量
classes_cnt = len(dataset.classes)

# 训练集数量
train_size = int(len(dataset) * train_split)
# 验证集数量
validation_size = int(len(dataset) * validation_split)
# 测试集数量
test_size = len(dataset) - train_size - validation_size

# 根据比例和随机种子划分，训练集，验证集，测试集
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size], 
                    generator=torch.Generator().manual_seed(random_seed))

# 训练集加载器，训练的时候调用
train_loader = DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)

# 验证集加载器，验证的时候调用
validation_loader = DataLoader(
    validation_dataset, batch_size=batch_size,
    num_workers=data_loading_workers, pin_memory=True,
)

if __name__ == '__main__':
    print(train_dataset.indices)
    print(validation_dataset.indices)
    print(test_dataset.indices)
    print(len(train_dataset), len(validation_dataset), len(test_dataset))