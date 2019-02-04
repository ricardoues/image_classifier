# image_classifier
Image Classifier Project from Data Scientist Nanodegree from Udacity

# Part 1: Developing an Image Classifier with Deep Learning
In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch. You can access to this with the following link: 

[Part 1](https://github.com/ricardoues/image_classifier/tree/master/Part1)


# Part 2:  Building the command line application
Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a pair of Python scripts that run from the command line. 

[Part 2](https://github.com/ricardoues/image_classifier/tree/master/Part2)

## How to run the Python scripts 

First, we will use /home/workspace/ImageClassifier as working directory.

**Example of usage**:

python train.py flowers --arch alexnet --learning_rate 0.001 --hidden_units 1500 --epochs 2 --gpu

python predict.py flowers/test/100/image_07896.jpg checkpoint.pth --top_k 10  --category_names cat_to_name.json   --gpu 
