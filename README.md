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

**Output**:

cuda is available
Epoch 1/2.. Train loss: 7.405.. Valid loss: 6.867.. Valid accuracy: 0.073
Epoch 1/2.. Train loss: 5.815.. Valid loss: 4.429.. Valid accuracy: 0.167
Epoch 1/2.. Train loss: 4.542.. Valid loss: 3.450.. Valid accuracy: 0.246
Epoch 1/2.. Train loss: 3.520.. Valid loss: 3.038.. Valid accuracy: 0.330
Epoch 1/2.. Train loss: 3.382.. Valid loss: 2.626.. Valid accuracy: 0.388
Epoch 1/2.. Train loss: 2.893.. Valid loss: 2.401.. Valid accuracy: 0.403
Epoch 1/2.. Train loss: 2.821.. Valid loss: 2.231.. Valid accuracy: 0.459
Epoch 1/2.. Train loss: 2.641.. Valid loss: 1.963.. Valid accuracy: 0.504
Epoch 1/2.. Train loss: 2.576.. Valid loss: 1.683.. Valid accuracy: 0.564
Epoch 1/2.. Train loss: 2.406.. Valid loss: 1.672.. Valid accuracy: 0.565
Epoch 1/2.. Train loss: 2.321.. Valid loss: 1.630.. Valid accuracy: 0.559
Epoch 1/2.. Train loss: 2.201.. Valid loss: 1.597.. Valid accuracy: 0.566
Epoch 1/2.. Train loss: 2.009.. Valid loss: 1.477.. Valid accuracy: 0.601
Epoch 1/2.. Train loss: 2.005.. Valid loss: 1.353.. Valid accuracy: 0.650
Epoch 1/2.. Train loss: 2.123.. Valid loss: 1.293.. Valid accuracy: 0.665
Epoch 1/2.. Train loss: 1.883.. Valid loss: 1.284.. Valid accuracy: 0.664
Epoch 1/2.. Train loss: 1.999.. Valid loss: 1.170.. Valid accuracy: 0.686
Epoch 1/2.. Train loss: 1.782.. Valid loss: 1.201.. Valid accuracy: 0.684
Epoch 1/2.. Train loss: 1.939.. Valid loss: 1.183.. Valid accuracy: 0.683
Epoch 1/2.. Train loss: 1.749.. Valid loss: 1.107.. Valid accuracy: 0.705
Epoch 2/2.. Train loss: 1.649.. Valid loss: 1.125.. Valid accuracy: 0.690
Epoch 2/2.. Train loss: 1.667.. Valid loss: 1.070.. Valid accuracy: 0.717
Epoch 2/2.. Train loss: 1.710.. Valid loss: 1.018.. Valid accuracy: 0.734
Epoch 2/2.. Train loss: 1.749.. Valid loss: 0.970.. Valid accuracy: 0.751
Epoch 2/2.. Train loss: 1.669.. Valid loss: 0.924.. Valid accuracy: 0.741
Epoch 2/2.. Train loss: 1.685.. Valid loss: 0.897.. Valid accuracy: 0.758
Epoch 2/2.. Train loss: 1.587.. Valid loss: 0.908.. Valid accuracy: 0.757
Epoch 2/2.. Train loss: 1.560.. Valid loss: 0.944.. Valid accuracy: 0.742
Epoch 2/2.. Train loss: 1.721.. Valid loss: 0.918.. Valid accuracy: 0.738
Epoch 2/2.. Train loss: 1.561.. Valid loss: 0.896.. Valid accuracy: 0.749
Epoch 2/2.. Train loss: 1.688.. Valid loss: 0.852.. Valid accuracy: 0.767
Epoch 2/2.. Train loss: 1.576.. Valid loss: 0.851.. Valid accuracy: 0.769
Epoch 2/2.. Train loss: 1.517.. Valid loss: 0.858.. Valid accuracy: 0.762
Epoch 2/2.. Train loss: 1.519.. Valid loss: 0.847.. Valid accuracy: 0.772
Epoch 2/2.. Train loss: 1.474.. Valid loss: 0.903.. Valid accuracy: 0.757
Epoch 2/2.. Train loss: 1.597.. Valid loss: 0.840.. Valid accuracy: 0.761
Epoch 2/2.. Train loss: 1.449.. Valid loss: 0.855.. Valid accuracy: 0.762
Epoch 2/2.. Train loss: 1.269.. Valid loss: 0.895.. Valid accuracy: 0.758
Epoch 2/2.. Train loss: 1.392.. Valid loss: 0.873.. Valid accuracy: 0.757
Epoch 2/2.. Train loss: 1.474.. Valid loss: 0.843.. Valid accuracy: 0.775
Epoch 2/2.. Train loss: 1.564.. Valid loss: 0.781.. Valid accuracy: 0.782
Test loss: 1.195.. Test accuracy: 0.678
Checkpoint saved


python predict.py flowers/test/100/image_07896.jpg checkpoint.pth --top_k 10  --category_names cat_to_name.json   --gpu 

**Output**:

cuda is available
['blanket flower', 'barbeton daisy', 'english marigold', 'sunflower', 'hibiscus', 'orange dahlia', 'watercress', 'black-eyed susan', 'gazania', 'mallow']
[0.8421755433082581, 0.08167364448308945, 0.03569231927394867, 0.0185739453881979, 0.005744044668972492, 0.004475531633943319, 0.0020602354779839516, 0.0012618767796084285, 0.0011671821121126413, 0.001093523926101625]

