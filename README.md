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
Epoch [1/10] Train Loss: 1.5022 | Test Loss: 1.1901 | Test Acc: 57.56%
Epoch [2/10] Train Loss: 1.0947 | Test Loss: 1.0629 | Test Acc: 62.96%
Epoch [3/10] Train Loss: 0.9050 | Test Loss: 0.8750 | Test Acc: 69.44%
Epoch [4/10] Train Loss: 0.7716 | Test Loss: 0.9236 | Test Acc: 67.89%
Epoch [5/10] Train Loss: 0.6762 | Test Loss: 0.8227 | Test Acc: 71.55%
Epoch [6/10] Train Loss: 0.5813 | Test Loss: 0.7836 | Test Acc: 74.34%
Epoch [7/10] Train Loss: 0.5088 | Test Loss: 0.8326 | Test Acc: 72.59%
Epoch [8/10] Train Loss: 0.4360 | Test Loss: 0.8158 | Test Acc: 73.66%
Epoch [9/10] Train Loss: 0.3700 | Test Loss: 0.8451 | Test Acc: 74.68%
Epoch [10/10] Train Loss: 0.3097 | Test Loss: 0.8841 | Test Acc: 74.19%


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
