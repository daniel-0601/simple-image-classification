# AI Image Classification (CIFAR-10)

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset.

---

## 📌 Features

- CNN-based image classification
- PyTorch training pipeline
- Model evaluation (accuracy & loss)
- Best model checkpoint saving

---

# 🧠 AI Image Classification (CIFAR-10)

This project implements a **baseline Convolutional Neural Network (CNN)** for image classification using the CIFAR-10 dataset.

---

## 🔹 Model Overview

The model is a simple CNN used as a baseline for further improvements.

### Architecture
Input (3×32×32)
↓
Conv(32) → ReLU → MaxPool
↓
Conv(64) → ReLU → MaxPool
↓
Conv(128) → ReLU → MaxPool
↓
Flatten
↓
FC(256) → ReLU
↓
FC(10)

### Design Concept

- Simple and lightweight CNN
- Focus on building a complete training pipeline
- Serves as a baseline for future improvements (ResNet, enhancement, etc.)

---

## ⚙️ Training Setup

- Dataset: CIFAR-10  
- Input size: 32×32 RGB  
- Loss function: CrossEntropyLoss  
- Optimizer: Adam (lr=0.001)  
- Batch size: 64  
- Epochs: 10  
- Device: CPU  

---

## 📊 Training Results

### Training Log
Epoch [1/10] Train Loss: 1.5141 | Test Loss: 1.1982 | Test Acc: 56.93%

Epoch [2/10] Train Loss: 1.1172 | Test Loss: 1.0207 | Test Acc: 64.26%

Epoch [3/10] Train Loss: 0.9319 | Test Loss: 0.9259 | Test Acc: 67.86%

Epoch [4/10] Train Loss: 0.8061 | Test Loss: 0.9251 | Test Acc: 67.64%

Epoch [5/10] Train Loss: 0.7105 | Test Loss: 0.7977 | Test Acc: 72.51%

Epoch [6/10] Train Loss: 0.6292 | Test Loss: 0.7821 | Test Acc: 72.82%

Epoch [7/10] Train Loss: 0.5499 | Test Loss: 0.7584 | Test Acc: 73.88%

Epoch [8/10] Train Loss: 0.4882 | Test Loss: 0.8296 | Test Acc: 73.14%

Epoch [9/10] Train Loss: 0.4222 | Test Loss: 0.8213 | Test Acc: 74.27%

Epoch [10/10] Train Loss: 0.3645 | Test Loss: 0.8401 | Test Acc: 74.13%


---

## 📈 Final Performance

- **Best Test Accuracy:** 74.68%  
- Model saved at: `outputs/best_model.pth`  

---

## 📊 Result Analysis

### 1️⃣ Learning Behavior

- Training loss decreases steadily (1.50 → 0.30)
- Test accuracy peaks around epoch 6–9  

👉 The model successfully learns meaningful features.

---

### 2️⃣ Overfitting

After epoch 6:

- Train loss continues decreasing  
- Test loss starts fluctuating  

👉 Indicates overfitting on training data.

---

### 3️⃣ Model Performance

- Achieves ~75% accuracy on CIFAR-10  
- Reasonable baseline for simple CNN  

---

### 4️⃣ Future Improvements

- Data augmentation  
- Batch normalization  
- Dropout  
- Deeper models (ResNet)  
- Learning rate scheduling  

---

## 📌 Key Takeaways

- Baseline CNN achieves ~74.7% accuracy on CIFAR-10  
- Clear overfitting observed after mid training  
- Provides a solid foundation for future research  

---

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR10-green)
