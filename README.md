# CIFAR-10 Image Classification using Convolutional Neural Networks
## Project Overview
This project implements an image classification model using Convolutional Neural Networks (CNNs) to classify images from the CIFAR-10 dataset. The model achieves an accuracy of 80% after training for 20 epochs.

## Table of Contents
-- Dataset
-- Model Architecture
-- Requirements
-- Installation
-- Training
-- Results

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Each class contains 6,000 images, with 50,000 training images and 10,000 test images

## Model Architecture
The CNN model consists of:

- Convolutional layers for feature extraction
- Max pooling layers for downsampling
- Fully connected layers for classification
- ReLU activation functions
- Batch normalization for improved training stability
