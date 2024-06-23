Image_Classification_Project

Image classification model development

Project Description:
This project involves building a Convolutional Neural Network (CNN) to classify images into six categories: buildings, forest, glacier, mountain, sea, and street.

Dataset:
* Train set: 14000 images
* Validation set: 7000 images
* Test set: 3000 images

Requirements:
* Python 3.x
* Git

Installation:
1. clone the repository:
  git clone https://github.com/mohammedshebinp/Image_Classifiction.git ,
  cd Image_Classifiction

3.  Install dependencies:
  pip install -r requirements.txt

4.  To train and evaluate the program:
  run python main.py

5.  To test the program:
  run python testmodel.py 

Model Architecture:
The model is built using TensorFlow and Keras and consists of the following layers:
* 4 Convolutional Layers
* 4 MaxPooling Layers
* 1 Flatten Layer
* 1 Dense Layer with ReLU activation
* 1 Dense Output Layer with Softmax activation

Hyperparameters
* Learning rate: 0.001
* Batch size: 32
* Epochs: 20

Data Preprocessing
* Images are resized to 150x150 pixels.
* Normalization is applied to scale pixel values to the range [0, 1].
* Evaluation

The model's performance is evaluated using the following metrics:
* Accuracy
* Precision
* Recall
