import os
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import StepLR
from dataset import train_loader, validation_loader, classes_cnt
from config import (epochs, out_dir, hub_repo, model_arch, momentum, batch_size,
                    print_freq, start_epoch, weight_decay, initial_learning_rate)

if __name__ == '__main__':

    # 确定训练所用的设备，是用gpu还是cpu，gpu要求是Nvidia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化迁移学习所用模型，从torch.hub加载预训练模型，修改源模型的全连接层，适配目标任务分类，这种手法叫微调。
    model = torch.hub.load(hub_repo, model_arch, pretrained=True)
    # resnext50_32x4d 最终汇聚成1000个分类，需要把这些分类，换成目标数据集分类数量，这个全连接层(fc)是我们要主要训练
    model.fc = nn.Linear(model.fc.in_features, classes_cnt) 
    # 使用xavier_uniform_方法为全连接层随机初始化参数
    nn.init.xavier_uniform_(model.fc.weight)
    # 将模型切换到指定训练设备
    model = model.to(device)

    # 定义随机梯度下降优化器，交叉熵损失函数，以及间隔学习率调整方法
    # params是找到模型中不属于fc的参数，这些是不参与训练的参数
    params = [param for name, param in model.named_parameters() 
                if name not in ["fc.weight", "fc.bias"]]
    # 定义随机梯度下降优化器实例，把要训练的全连接层赋予高一点的学习率，10倍，传递初始学习率，动量，和衰减
    optimizer = torch.optim.SGD([{'params': params,}, {'params': model.fc.parameters(), 'lr': initial_learning_rate * 10}], 
                                lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)
    # 定义交叉熵损失函数，并切换到指定设备
    criterion = nn.CrossEntropyLoss().to(device)
    # 定义间隔学习率调整实例，设置step_size 30 gamma 0.1
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # 迭代循环
    for epoch in range(start_epoch, epochs):
        print("--------------------------TRAINING--------------------------------")
        model.train() # 将模型切换到训练模式     

        # 训练循环，从train_loader中按照batch_size的设置，按组抽取图像tensor和其标签
        for i, (images, labels) in enumerate(train_loader):
            # 一组图像有batch_size个tensor，shape([batch_size, 3, image_size, image_size])
            # to(device) 切换至当前训练设备
            images = images.to(device)
            # 一组标签有batch_size个tensor，shape([batch_size])
            # to(device) 切换至当前训练设备
            labels = labels.to(device)

            # 把图像代入模型，进行一次前向传播，得到猜想标签，shape([batch_size, classes_cnt])
            output = model(images)
            # 根据损失函数，计算猜想值（概率）与实际值（概率）之间的距离
            loss = criterion(output, labels)

            # 计算准确率，output有batch_size组长度为classes_cnt的数组，
            # argmax可以找到每组中值最大的那一个的索引，即预测的标签，shape([batch_size])
            prediction = torch.argmax(output, dim=1)
            # 计算预测标签shape([batch_size])与实际标签shape([batch_size])相匹配的总数有多少个，得到一个整数
            train_correct = (prediction == labels).sum()
            # 根据匹配数量与batch_size的比值得到训练准确率。
            train_acc = (train_correct.float()) / images.shape[0]

            if i % print_freq == 0:
                print("[Epoch {:02d}] ({:03d}/{:03d}) | Loss: {:.18f} | ACC: {:.2f} %".format(epoch, i + 1, len(train_loader), loss.item(), train_acc * 100))
            
            # 反向传播，更新模型
            # 首先梯度置零
            optimizer.zero_grad()
            # 然后反向传播
            loss.backward()
            # 最后执行优化器，更新模型参数
            optimizer.step()

        print("--------------------------VALIDATION-----------------------------")
        model.eval() # 将模型切换到验证模式     

        # 这一个with表示，with里的步骤全都与梯度无关，不参与梯度下降
        with torch.no_grad():
            # 验证循环，从validation_loader中按照batch_size的设置，按组抽取图像tensor和其标签
            for i, (images, labels) in enumerate(validation_loader):
                # 一组图像有batch_size个tensor，shape([batch_size, 3, image_size, image_size])
                # to(device) 切换至当前训练设备
                images = images.to(device)
                # 一组标签有batch_size个tensor，shape([batch_size])
                # to(device) 切换至当前训练设备
                labels = labels.to(device)

                # 对验证集中的图像进行预测，得到预测输出，shape([batch_size, classes_cnt])
                output = model(images)
                # 根据损失函数，计算验证损失
                loss = criterion(output, labels)

                # 同训练循环相应部分
                prediction = torch.argmax(output, dim=1)
                val_correct = (prediction == labels).sum()
                val_acc = (val_correct.float()) / images.shape[0]
                
                if i % print_freq == 0:
                    print("[Epoch {:02d}] ({:03d}/{:03d}) | Loss: {:.18f} | ACC: {:.2f} %".format(epoch, i + 1, len(validation_loader), loss.item(), val_acc * 100))

        # 这一代训练结束，执行学习率调整
        scheduler.step()

        # 保存模型存档
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, "checkpoint.pth")
        state = {
            'epoch': epoch + 1,
            'arch': model_arch,
            'hub_repo': hub_repo,
            'state_dict': model.state_dict(),
            'classes_cnt': classes_cnt
        }

        torch.save(state, filename)