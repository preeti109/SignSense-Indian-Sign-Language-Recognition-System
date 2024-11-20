
# SignSense-Indian Sign Language Recognition System

SignSense is a machine learning-based system designed to recognize gestures in Indian Sign Language (ISL). This project aims to analyze and identify various alphabets and numbers using a robust dataset of hand gesture images, helping bridge the communication gap for individuals with speech and hearing impairments.

Introduction to Sign Languages

Sign Languages are a group of languages that use predefined actions and movements to convey messages. These languages are primarily developed to aid deaf and verbally challenged individuals. Sign languages rely on a precise and simultaneous combination of:
    Hand movements
    Hand orientation
    Hand shapes

Different regions have developed distinct sign languages, such as American Sign Language (ASL) and Indian Sign Language (ISL). In this project, we focus on Indian Sign Language, a prevalent sign language in South Asian countries.

Prerequisites

     Dataset 

     Python 

     OpenCV

     Keras

     Tensorflow

     Numpy

     Keras

     Tensorflow

Objective

This project aims to analyze and recognize various alphabets and numbers from a dataset of Indian Sign Language images. The system is designed to work with a diverse dataset that includes:

     Images captured under different lighting conditions
     Various hand orientations and shapes

By training on such a divergent dataset, the project achieves robust performance in recognizing ISL gestures with high accuracy.     

Execution

Collect Data: Gather a dataset containing images of Indian Sign Language gestures. Ensure the dataset includes diverse lighting conditions, hand orientations, and shapes for better generalization.

Split Data: Divide the dataset into two subsets: training and validation. The training set is used to train the model, the validation set to tune hyperparameters.

Preprocess Data: Preprocess the images by resizing them to a uniform size, normalizing pixel values for consistent input, and applying data augmentation techniques like flipping, rotation, or zoom to increase the dataset’s diversity.

Build Model: Design a Convolutional Neural Network (CNN) architecture tailored for gesture recognition. Include convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.
![1](https://github.com/user-attachments/assets/41d47518-424c-4b7b-b2f7-c9975107504e | width=100)

Train Model: Train the CNN using the prepared training data. Optimize the model's weights through backpropagation and evaluate its performance on the validation set. Save the trained model for future use.

Real-Time Recognition: Implement real-time gesture recognition by integrating a webcam or video input. Process each frame using the trained model to identify and display the recognized gesture in real-time.


## Group Members

- [@Anuradha Bansode](https://github.com/anyalisis12)
- [@Sayali Tachale](https://github.com/Sayali2408)
- [@Preeti Dubile](https://github.com/preeti109)

