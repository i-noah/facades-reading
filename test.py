import torch
from dataset import test_dataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(r'.\out\checkpoint.pth')

    hub_repo = checkpoint['hub_repo']
    arch = checkpoint['arch']
    classes_cnt = checkpoint['classes_cnt']
    model = torch.hub.load(hub_repo, arch, pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, classes_cnt)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    model.eval()
    check = 0

    for image, target in test_dataset:
        image = image.to(device).unsqueeze(0)
        output = model(image)

        prediction = torch.argmax(output, dim=1).cpu().item()
        if prediction == target:
            check += 1
    
    print("{} %".format((check / len(test_dataset)) * 100))