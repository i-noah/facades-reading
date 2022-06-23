import os
import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import StepLR
from dataset import train_loader, validation_loader, classes_cnt
from config import (epochs, out_dir, hub_repo, model_arch, momentum, batch_size,
                    print_freq, start_epoch, weight_decay, initial_learning_rate)

if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)
    
    nn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.hub.load(hub_repo, model_arch, pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, classes_cnt)
    nn.init.xavier_uniform_(model.fc.weight)
    model = model.to(device)

    params = [param for name, param in model.named_parameters() 
                if name not in ["fc.weight", "fc.bias"]]

    optimizer = torch.optim.SGD([{'params': params,}, {'params': model.fc.parameters(), 'lr': initial_learning_rate * 10}], 
                                lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(start_epoch, epochs):
        print("--------------------------TRAINING--------------------------------")
        model.train()       

        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            prediction = torch.max(output, 1)[1]
            train_correct = (prediction == target).sum()
            train_acc = (train_correct.float()) / batch_size

            if i % print_freq == 0:
                print("[Epoch {:02d}] ({:03d}/{:03d}) | Loss: {:.18f} | ACC: {:.2f} %".format(epoch, i, len(train_loader), loss.item(), train_acc * 100))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("--------------------------VALIDATION-----------------------------")
        model.eval()

        with torch.no_grad():
            for i, (images, target) in enumerate(validation_loader):
                images = images.to(device)
                target = target.to(device)
                output = model(images)
                loss = criterion(output, target)

                prediction = torch.max(output, 1)[1]
                val_correct = (prediction == target).sum()
                val_acc = (val_correct.float()) / batch_size
                
                if i % print_freq == 0:
                    print("[Epoch {:02d}] ({:03d}/{:03d}) | Loss: {:.18f} | ACC: {:.2f} %".format(epoch, i, len(validation_loader), loss.item(), val_acc * 100))

        scheduler.step()

        filename = os.path.join(out_dir, "checkpoint.pth")
        state = {
            'epoch': epoch + 1,
            'arch': model_arch,
            'hub_repo': hub_repo,
            'state_dict': model.state_dict(),
            'classes_cnt': classes_cnt
        }

        torch.save(state, filename)