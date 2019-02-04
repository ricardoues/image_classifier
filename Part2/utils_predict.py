import numpy as np
from PIL import Image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    size = 256, 256
    image.thumbnail(size)
    
    width, height = image.size # Get dimensions

    new_width = 224
    new_height = 224

    left = (width - new_width)/2.
    top = (height - new_height)/2.
    right = (width + new_width)/2.
    bottom = (height + new_height)/2.

    image = image.crop((left, top, right, bottom))
    
    image = np.array(image)
    image =  image.astype(np.float64)
        
    # Scaling the values to the range 0-1.     
    
    min_values = np.array([0, 0, 0])
    max_values = np.array([255, 255, 255])    
    
    # Using /= and *= allows you to eliminate an intermediate temporary array, 
    # thus saving some memory. 
    # https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    
    image -= min_values 
    image /= max_values - min_values
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    image -= means 
    image /= stds 
        
    # we make the current last axis, and make it the first axis.
    image = image.transpose(2, 0, 1)
    
    
    return image 
    

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


def recover_key(d, value):
    for k, v in d.items():
        if v == value: 
            return k

        