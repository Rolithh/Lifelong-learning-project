import torch
import torchvision.models as models

def load_model():
    """Load AlexNet model for regression"""
    
    #model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model = models.alexnet(pretrained=True)
    
    # Add one input channel to the first layer
    with torch.no_grad():
        weight = model.features[0].weight.data
        bias = model.features[0].bias.data
    
        layer = torch.nn.Conv2d(4, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2))
        layer.weight.data = torch.cat((weight, weight[:,2:,:,:]), 1)
        layer.bias.data = bias
        model.features[0] = layer
    
    # Change classifier module (reset linear layers + change last layer)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(9216, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(4096, 5)
    )
    
    print(model)
    return model


def minMSELoss(pred, gt, device=None):
    """loss for training"""

    if device is not None:
        loss = torch.zeros((len(gt)), device=device)
    else:
        loss = torch.zeros((len(gt)))

    for k in range(len(gt)):
        
        mse = torch.sum((pred[k] - gt[k])**2, -1)
        loss[k] = torch.min(mse)
        
        
    return loss.mean()
        
        