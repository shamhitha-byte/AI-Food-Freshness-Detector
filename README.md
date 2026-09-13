# AI Food Freshness Detector

## Project Overview

AI Food Freshness Detector is a deep learning project that classifies food images as **Fresh** or **Rotten**.

The project uses **MobileNetV2 transfer learning** and provides a simple Streamlit web application for food freshness prediction.

## Features

- Fresh or Rotten food classification
- MobileNetV2-based deep learning model
- Image preprocessing and data augmentation
- Prediction confidence score
- Streamlit web interface
- Model evaluation and performance graphs

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- Streamlit

## Model

The project uses **MobileNetV2** with ImageNet pretrained weights.

Transfer learning is used to extract useful features from food images and classify them into two classes:

- Fresh
- Rotten

## Results

**Test Accuracy: 83.33%**

The model correctly classified **20 out of 24 test images**.

### Confusion Matrix

```text
[[11, 1],
 [ 3, 9]]
