"""
A neuron has certain inputs, inputs are multiplied by weights.
Each input is multiplied by their corresponding weights and is summed up.
The total is sent as the input to the activation function (sigmoid here).
The output of the sigmoid function is the final output (always between 0 and 1).
"""


import numpy as np

def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        result = np.dot(self.weights, inputs) + self.bias
        return sigmoid(result)


weights = np.array([0, 1])
bias = 4
n = Neuron(weights, bias)

x = np.array([2, 3])

#print(n.feedforward(x))
