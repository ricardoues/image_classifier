# TODO: Build your network
# We define a function in order to build the model. 

def build_dl_model(arch, hidden_units):
    
    import matplotlib.pyplot as plt 
    import os 
    import numpy as np
    
    import torch 
    from torch import nn 
    from torch import optim
    import torch.nn.functional as F 
    from torchvision import datasets, transforms, models   
    import json
    
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
    elif arch == "alexnet": 
        model = models.alexnet(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True) 
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True) 
    elif arch == "inception_v3": 
        model = models.inception_v3(pretrained=True) 
    else:
        model = models.vgg19(pretrained=True)
                   
    # Freeze parameters 
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict 

    # I tried different architectures and hyperparameters but the 
    # following works well. 

    classifier = nn.Sequential(OrderedDict([
                                           ('fc1', nn.Linear(25088, hidden_units)) , 
                                           ('relu1', nn.ReLU()), 
                                           ('dropout1', nn.Dropout(0.4)), 
                                           ('fc2', nn.Linear(hidden_units, 102)),
                                           ('output', nn.LogSoftmax(dim=1))
                                           ]))

    model.classifier = classifier 
    
    return model 
        
 