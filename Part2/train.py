import argparse 
import functions_train
from utils_train import load_data 
from functions_train import build_dl_model 

import matplotlib.pyplot as plt 
import os 
import numpy as np
import sys
    
import torch 
from torch import nn 
from torch import optim
import torch.nn.functional as F 
from torchvision import datasets, transforms, models   
import json


parser = argparse.ArgumentParser(description='Training a deep learning model for classifying flowers')

parser.add_argument('data_directory', action="store")
parser.add_argument('--save_dir', action="store", dest="save_dir", default="." ) 
parser.add_argument('--arch', action="store", dest="arch", default="vgg19") 
parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.01) 
parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=1500)
parser.add_argument('--epochs', action="store", dest="epochs", default=5)
parser.add_argument('--gpu', action="store_true", dest="gpu", default=False)

results = parser.parse_args()


data_directory = results.data_directory
save_dir = results.save_dir
arch = results.arch
learning_rate = results.learning_rate 
hidden_units = results.hidden_units 
epochs = results.epochs 
gpu  = results.gpu
learning_rate = float(learning_rate)
hidden_units = int(float(hidden_units))
epochs = int(float(epochs))


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if gpu and  not torch.cuda.is_available() :
    print("There is no a gpu device available")
    sys.exit()
    
   
trainloader, validloader, testloader, class_to_idx = load_data(data_directory)

model = build_dl_model(arch, hidden_units)


message_cuda = "cuda is available" if torch.cuda.is_available() else "cuda is not available"

print(message_cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.NLLLoss()

# Only train the classifier parameters 
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

# We move our model to cuda, if this is available.
model.to(device);


steps = 0
running_loss = 0
print_every = 5


for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
            
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                  
                
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "    
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
            
# TODO: Do validation on the test set

test_loss = 0 
accuracy = 0 

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
                    
        test_loss += batch_loss.item()
                    
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        
print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {accuracy/len(testloader):.3f}")
                              
# TODO: Save the checkpoint 


checkpoint = { 'arch' : arch ,
               'state_dict' : model.state_dict(),
               'class_to_idx' : class_to_idx, 
               'hidden_units': hidden_units               
             }

if save_dir.endswith('/'):
    torch.save(checkpoint, save_dir + 'checkpoint.pth')
else:
    torch.save(checkpoint, save_dir + '/' + 'checkpoint.pth')
    
print("Checkpoint saved")    
    
    
