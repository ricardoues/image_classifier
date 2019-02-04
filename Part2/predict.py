import argparse
from PIL import Image
from functions_predict import load_checkpoint
from functions_predict import predict 
from utils_predict import recover_key
import json
import sys 
import torch

parser = argparse.ArgumentParser(description='Predicting a flower name from an image')

parser.add_argument('path_to_image', action="store")
parser.add_argument('checkpoint', action="store")
parser.add_argument('--top_k', action="store", dest="top_k", default=3)
parser.add_argument('--category_names', action="store", dest="category_names", default="")
parser.add_argument('--gpu', action="store_true", dest="gpu", default=False)


results = parser.parse_args()

path_to_image = results.path_to_image
checkpoint = results.checkpoint
top_k = results.top_k
category_names = results.category_names
gpu = results.gpu
top_k = int(float(top_k))


if gpu and  not torch.cuda.is_available() :
    print("There is no a gpu device available")
    sys.exit()

message_cuda = "cuda is available" if torch.cuda.is_available() else "cuda is not available"

print(message_cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model = load_checkpoint(checkpoint)

model.to(device); 

class_to_idx = model.class_to_idx 

probs, classes = predict(path_to_image, model, top_k) 

if not category_names:
    print(classes) 
    print(probs) 
    
else:
    
    name_classes = []
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)    
    
    for i in classes:
        flower_key = recover_key(class_to_idx, i)
        name_classes.append(cat_to_name[flower_key])
    
    print(name_classes)
    print(probs)

