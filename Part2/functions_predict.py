import torch
from functions_train import build_dl_model 
from PIL import Image
from utils_predict import process_image 

def load_checkpoint(filepath, arch, hidden_units):
    checkpoint = torch.load(filepath)
    
    model = build_dl_model(arch, hidden_units)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model 


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
    im = Image.open(image_path)
    im = process_image(im)
    im = torch.from_numpy(im)
    
    # This web page was very useful
    # https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612/3
    im.unsqueeze_(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # This web page was very useful, in order to solve a problem
    #https://discuss.pytorch.org/t/input-type-torch-cuda-doubletensor-and-weight-type-torch-cuda-floattensor-should-be-the-same/22704/2
    im = im.to(device, dtype=torch.float)
    model.to(device); 
    
    ps = torch.exp(model(im))    
    top_p, top_class = ps.topk(topk, dim=1)
    
    
    # This post was very helpful 
    # https://discuss.pytorch.org/t/convert-to-numpy-cuda-variable/499/10
    top_p = top_p.cpu().detach().numpy()
    top_p = top_p.flatten()    
    top_p = top_p.tolist()
    
    top_class = top_class.cpu().detach().numpy()    
    top_class = top_class.flatten()
    top_class = top_class.tolist()
    
    return top_p, top_class 