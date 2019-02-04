# Load the data

def load_data(data_dir):
        
    import matplotlib.pyplot as plt 
    import os 
    import numpy as np
    
    import torch 
    from torch import nn 
    from torch import optim
    import torch.nn.functional as F 
    from torchvision import datasets, transforms, models   
    import json
    
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                           transforms.RandomResizedCrop(224), 
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(), 
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])
                                          ])
    
    
    valid_transforms = transforms.Compose([transforms.Resize(255), 
                                           transforms.CenterCrop(224), 
                                           transforms.ToTensor(), 
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    test_data  = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    test_data  = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    # In order to treat with the problem of imbalanced classes we use a PyTorch sampler.
    from sampler import ImbalancedDatasetSampler

    batch_size = 64

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_data) )

    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        

    class_to_idx = train_data.class_to_idx

    
    return trainloader, validloader, testloader, class_to_idx 
    
