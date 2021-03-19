# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:15:13 2021

@author: Laetitia Haye
"""

import torch
import copy


def load_model():
    """Load AlexNet model for regression"""
    
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

    
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

    return model

#a_model = load_model()
#print(a_model)

def create_new_layer(model):
    """Adds a new task-specific layer to the model. New parameters are randomly\
    initialized"""
    
    model_copy = copy.deepcopy(model)
    number_of_neurones = model_copy.classifier[-1].out_features
    
    # Extend the last layer of the classifier module
    model_copy.classifier[-1] = torch.nn.Linear(4096, number_of_neurones + 5)
   
    #TO DO : random initialization (of the 5 last neurones only)
    
    return model_copy



def minMSELoss(pred, gt):
    """loss for training"""
    
    loss = torch.zeros((len(gt)))
    for k in range(len(gt)):
        
        mse = torch.sum((pred[k] - gt[k])**2, -1)
        loss[k] = torch.min(mse)
        
        
    return loss


