
# Cats vs Dogs Classifier

This repository contains a TensorFlow-based neural network model designed to classify images as either cats or dogs. The model is built using the TensorFlow and TensorFlow Datasets libraries and trained on the `cats_vs_dogs` dataset available through TensorFlow Datasets.

## Requirements

- Python 3.x
- TensorFlow 2.x
- TensorFlow Datasets
- Matplotlib

## Dataset

The dataset used is `cats_vs_dogs` from TensorFlow Datasets. It is automatically downloaded and loaded by the script. The dataset is split into 80% for training and 20% for validation.

## Model Architecture

The model is a Sequential model comprising the following layers:

- Conv2D: 32 filters, kernel size (3,3), activation 'relu', input shape (128, 128, 3)
- MaxPooling2D: pool size (2,2)
- Conv2D: 64 filters, kernel size (3,3), activation 'relu'
- MaxPooling2D: pool size (2,2)
- Flatten
- Dense: 128 units, activation 'relu'
- Dropout: 0.5
- Dense: 2 units, activation 'softmax'

## Preprocessing

Images are resized to 128x128 pixels and normalized to have values between 0 and 1.

## Training

The model is compiled with the Adam optimizer, using `sparse_categorical_crossentropy` as the loss function and tracking `accuracy` as the metric. It is trained for 10 epochs.

## Visualization

Training accuracy and validation accuracy are plotted against epochs to visualize the model's performance over time.

## Saving the Model

The trained model is saved to `saved_models/cats_vs_dogs_model.h5`.

## Usage

To train the model, simply run `main.py`. Ensure you have the required libraries installed.
