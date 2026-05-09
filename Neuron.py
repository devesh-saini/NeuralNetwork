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

        # Cached values from the last forward pass — needed for backprop
        self.last_input = None
        self.last_z = None
        self.last_a = None

    def feedforward(self, inputs):
        self.last_input = inputs
        self.last_z = np.dot(self.weights, inputs) + self.bias
        self.last_a = sigmoid(self.last_z)
        return self.last_a

    def apply_gradients(self, dL_dw, dL_db, learning_rate):
        self.weights -= learning_rate * dL_dw
        self.bias    -= learning_rate * dL_db