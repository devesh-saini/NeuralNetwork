# Neural Network from Scratch

**A Pure NumPy Implementation of Neural Networks**  
Built to understand Deep Learning fundamentals from the ground up.

---

## ✨ Overview

This project implements a complete **Multi-Layer Neural Network** without using any high-level machine learning frameworks like TensorFlow, PyTorch, or scikit-learn.

The goal was to deeply understand the core mechanics of deep learning:
- Forward Propagation
- Backward Propagation (Backpropagation)
- Weight Initialization
- Activation Functions
- Loss Computation & Optimization
- Gradient Flow

This foundational knowledge has proven extremely valuable when working with LLMs, fine-tuning, and building complex AI systems.

---

## 🎯 Key Features

- Implemented entirely with **NumPy**
- Supports arbitrary number of hidden layers
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Binary & Multi-class classification support
- Proper weight initialization (Xavier/He)
- Gradient descent optimization
- Loss visualization and training metrics
- Clean, well-documented, modular code

---

## 🚀 Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/devesh-saini/NeuralNetwork.git
cd NeuralNetwork
```
### 2. Install Dependencies
```bash
pip install numpy matplotlib seaborn
```
### 3. Run the Examples
```bash
# Train on a sample dataset (binary classification)
python main.py

# Or explore different experiments
python experiments/xor.py
python experiments/mnist_digits.py
```

## 🧠 What I Learned

- How neural networks actually work under the hood
- Mathematics behind backpropagation and chain rule
- Importance of proper weight initialization
- Impact of different activation functions
- Debugging gradient issues (vanishing/exploding gradients)
- Building intuition that helps with modern LLM fine-tuning

This project strengthened my ability to debug, optimize, and innovate in production AI systems.
