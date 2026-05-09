import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights, dtype=float)
        self.bias = float(bias)
        # Stored during forward pass, needed for backprop
        self.last_z = None
        self.last_input = None

    def feedforward(self, inputs):
        self.last_input = np.array(inputs, dtype=float)
        self.last_z = np.dot(self.weights, inputs) + self.bias
        return sigmoid(self.last_z)

    def update(self, dL_dw, dL_db, lr):
        self.weights -= lr * dL_dw
        self.bias    -= lr * dL_db