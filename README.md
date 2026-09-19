# 🍎 AI Food Freshness Detector

An AI-based web application that uses Deep Learning to classify food images as **Fresh** or **Rotten**.

## 📌 Project Overview

This project uses a **MobileNetV2-based Convolutional Neural Network (CNN)** to detect the freshness of food from an uploaded image.

The application is developed using **Python and Streamlit**.

## 🧠 Model

- Model: MobileNetV2
- Transfer Learning: Yes
- Image Size: 160 × 160
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Classes:
  - Fresh
  - Rotten

## 📊 Dataset

The project uses the **Fruits and Vegetables Dataset** containing approximately **12,000 images**.

The dataset contains fresh and rotten categories of:

- Apple
- Banana
- Mango
- Orange
- Strawberry
- Bell Pepper
- Carrot
- Cucumber
- Potato
- Tomato

The dataset is divided into:

- Training set
- Validation set
- Test set

## 🎯 Model Performance

| Metric | Result |
|---|---:|
| Test Images | 1,815 |
| Test Accuracy | **95.32%** |
| Test Loss | **0.1291** |
| Correct Predictions | 1,730 |
| Incorrect Predictions | 85 |

### Confusion Matrix

```text
                 Predicted
              Fresh   Rotten

Actual Fresh    890      36
Actual Rotten    49     840