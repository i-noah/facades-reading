import torch

if __name__ == '__main__':
    checkpoint = torch.load('./out/checkpoint.pth')

    hub_repo = checkpoint['hub_repo']
    arch = checkpoint['arch']
    classes_cnt = checkpoint['classes_cnt']
    model = torch.hub.load(hub_repo, arch, pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, classes_cnt)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(3, 224, 224, requires_grad=True).unsqueeze(0)

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "./out/ImageClassifier.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')