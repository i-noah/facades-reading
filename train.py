import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
import torch.utils.data.distributed
from dataset import train_loader, valid_loader

hub_repo = 'pytorch/vision:v0.10.0'
arch = 'resnext101_32x8d'
out_dir = './out'
momentum = 0.9
initial_learning_rate = 5e-5
weight_decay = 1e-3
start_epoch = 0
epochs = 10
print_freq = 10

if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)

    model = torch.hub.load(hub_repo, arch, pretrained=True)
    
    model.fc = nn.Linear(model.fc.in_features, 2)
    nn.init.xavier_uniform_(model.fc.weight)

    if torch.cuda.is_available():
        model.cuda()

    params = [param for name, param in model.named_parameters() 
                if name not in ["fc.weight", "fc.bias"]]

    # define loss function (criterion), optimizer, and learning rate scheduler  
    optimizer = torch.optim.SGD([{'params': params,}, {'params': model.fc.parameters(), 'lr': initial_learning_rate * 10}], 
                                lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss().cuda()

    # Sets the learning rate to the initial LR decayed by 10 every 30 epoch
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(start_epoch, epochs):
        # switch to train mode and train for one epoch
        model.train()
        for i, (images, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            if i % print_freq == 0:
                print("[Train Epoch {}] ({}/{}) Loss {}".format(epoch, i, len(train_loader), loss.item()))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # switch to evaluate mode and evaluate on validation set
        model.eval()

        with torch.no_grad():
            for i, (images, target) in enumerate(valid_loader):
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                output = model(images)
                loss = criterion(output, target)

                if i % print_freq == 0:
                    print("[Test  Epoch {}] ({}/{}) Loss {}".format(epoch, i, len(valid_loader), loss.item()))
        
        scheduler.step()

        filename = os.path.join(out_dir, "checkpoint.pth.tar")
        torch.save({
            'epoch': epoch + 1,
            'arch': arch,
            'hub_repo': hub_repo,
            'state_dict': model.state_dict()
        }, filename)