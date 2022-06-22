import torch
from dataset import test_dataset

if __name__ == '__main__':
    checkpoint = torch.load(r'.\out\checkpoint.pth')

    hub_repo = checkpoint['hub_repo']
    arch = checkpoint['arch']
    model = torch.hub.load(hub_repo, arch, pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    check = 0

    for image, target in test_dataset:
        if torch.cuda.is_available():
            image = image.cuda(non_blocking=True)
        image = image.unsqueeze(0)
        output = model(image)

        prediction = torch.argmax(output, dim=1).cpu().item()
        if prediction == target:
            check += 1
    
    print("{} %".format((check / len(test_dataset)) * 100))