# CIFAR-10-Image-Classification-with-Convolutional-Neural-Network-CNN-
This repository contains code for training and evaluating a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using TensorFlow/Keras.
Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

Prerequisites
Make sure you have the following dependencies installed:

Python (>=3.6)
TensorFlow (>=2.0)
Keras
Matplotlib
NumPy
Scikit-learn
Installation
Clone this repository: git clone https://github.com/anamali153/cifar10-cnn.git

Usage
Run train.py to train the model.
Run evaluate.py to evaluate the trained model on the test set.
You can also visualize sample images from the dataset using visualize_dataset.py.

Model Architecture:
The model architecture consists of the following layers:
Convolutional layer with 32 filters and a kernel size of (3,3), using ReLU activation.
MaxPooling layer with a pool size of (3,3).
Flatten the layer to flatten the output of the previous layer.
Fully connected Dense layer with 256 units and ReLU activation.
Output Dense layer with 10 units (equal to the number of classes) and softmax activation.
The model is compiled using the Adam optimizer with a learning rate of 0.001 and categorical cross-entropy loss.
Results
The model achieves an accuracy of 0.67 on the test set.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
CIFAR-10 dataset




