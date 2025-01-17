#Forest Fire Detection Using Satellite Imagery
Introduction

This project aims to develop a system that uses computer vision, specifically Convolutional Neural Networks (CNNs), to automatically detect wildfires in satellite images of forests. The system will help monitor large areas and provide early warnings about potential wildfire threats, improving response time and potentially saving lives and property.
Objectives

    Wildfire Detection: The primary goal is to train a machine learning model to recognize wildfire images from satellite photos.
    User Interface (GUI): Develop a simple GUI using Tkinter that allows users to upload an image and get a prediction of whether the image contains a wildfire or not.
    Automation for Safety: Automating wildfire detection helps improve the speed of response in case of a wildfire, enhancing the safety of forested areas.

Dataset

The model is trained using the Wildfire Prediction Dataset available on Kaggle. This dataset contains satellite images of forests, categorized into two classes:

    Wildfire: Images with visible signs of a fire.
    No Wildfire: Images of forests without fires.

You can download the dataset from Kaggle:
Wildfire Prediction Dataset
Dataset Structure

The dataset includes three main folders:

    train: Contains images for training the model.
    valid: Contains images for validating the model during training.
    test: Contains images for testing the model after training.

Each folder has two subfolders:

    Wildfire: Images with wildfires.
    NoWildfire: Images without wildfires.

Steps
Step 1: Installing Required Libraries

Make sure to install the necessary libraries before running the code:

pip install tensorflow keras pillow gradio kagglehub

Step 2: Loading and Preprocessing the Data

We use TensorFlow’s ImageDataGenerator to preprocess the dataset. This tool handles tasks like resizing images and rescaling pixel values to prepare them for input into the model.
Step 3: Building the CNN Model

The model consists of convolutional layers for feature extraction followed by dense layers for classification. It is trained on the dataset and saves the trained model as ffd_model.h5.
Step 4: Training the Model

The model is trained using the training and validation data for a specified number of epochs. After training, the model is saved to disk.

history = model.fit(train_generator, validation_data=valid_generator, epochs=10, verbose=1)
model.save("ffd_model.h5")

Step 5: Building the GUI with Tkinter

We then create a simple GUI using Tkinter to upload images and get predictions.

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("ffd_model.h5")

Step 6: Testing the Model

Once the GUI is set up, the model can be tested by selecting an image. The system will predict whether the image contains a wildfire or not.
Conclusion

This project demonstrates the power of machine learning in forest fire detection, providing an automated solution to assist in monitoring large areas for potential wildfires. By utilizing computer vision and deep learning, the system can accurately differentiate between images with and without fires, potentially saving valuable time during wildfire events.
Future Enhancements

    Real-Time Monitoring: Integrating real-time satellite imagery feeds for continuous wildfire detection.
    Improved Model Accuracy: Fine-tuning the model for better performance, especially on edge cases.
    Mobile App Integration: Developing a mobile app to allow users to upload images directly from their phones.
