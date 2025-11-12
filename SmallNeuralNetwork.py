import numpy as np
from Neuron import Neuron

"""
Input Layer -> hidden Layer -> output layer -> output
"""

class NeuralNetwork:
    def __init__(self):
        weights = np.array([0,1,2,3,4])
        bias = 0

        self.hidden1 = Neuron(weights, bias)
        self.hidden2 = Neuron(weights, bias)
        self.hidden3 = Neuron(weights, bias)
        self.hidden4 = Neuron(weights, bias)
        self.hidden5 = Neuron(weights, bias)
        self.output = Neuron(weights, bias)

    def feedforward(self, input):

        outputHidden1 = self.hidden1.feedforward(input)
        outputHidden2 = self.hidden2.feedforward(input)
        outputHidden3 = self.hidden2.feedforward(input)
        outputHidden4 = self.hidden2.feedforward(input)
        outputHidden5 = self.hidden2.feedforward(input)

        outputOutput = self.output.feedforward(np.array([outputHidden1, outputHidden2, outputHidden3,
                                                         outputHidden4, outputHidden5]))

        return outputOutput


network = NeuralNetwork()
inputs = np.array([2,3,4,5,6])
print(network.feedforward(inputs))
