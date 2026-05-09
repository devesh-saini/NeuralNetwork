import numpy as np
from Neuron import Neuron, sigmoid, sigmoid_derivative

def random_weights(n_in):
    limit = np.sqrt(6 / (n_in + 1))
    return np.random.uniform(-limit, limit, size=n_in)

def binary_cross_entropy(y_pred, y_true):
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class NeuralNetwork:
    def __init__(self, n_inputs=5, n_hidden=5, learning_rate=0.1):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.lr = learning_rate

        self.hidden_neurons = [
            Neuron(random_weights(n_inputs), bias=np.random.randn() * 0.1)
            for _ in range(n_hidden)
        ]
        self.output_neuron = Neuron(random_weights(n_hidden), bias=np.random.randn() * 0.1)

    def feedforward(self, inputs):
        self.hidden_outputs = np.array([
            n.feedforward(inputs) for n in self.hidden_neurons
        ])
        self.prediction = self.output_neuron.feedforward(self.hidden_outputs)
        return self.prediction

    def backprop(self, y_true):
        y_pred = self.prediction

        # ------------------------------------------------------------------ #
        # OUTPUT LAYER
        # dL/da_out  — derivative of BCE loss w.r.t. output activation
        # dL/dz_out  — chain through sigmoid: multiply by sigmoid'(z_out)
        # dL/dW_out  — each weight's gradient = dL/dz_out * the input it multiplied
        # dL/db_out  — bias gradient = dL/dz_out (bias has implicit input of 1)
        # ------------------------------------------------------------------ #
        y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
        dL_da_out = -(y_true / y_pred_clipped) + (1 - y_true) / (1 - y_pred_clipped)
        dL_dz_out = dL_da_out * sigmoid_derivative(self.output_neuron.last_z)

        dL_dW_out = dL_dz_out * self.hidden_outputs   # shape: (n_hidden,)
        dL_db_out = dL_dz_out                          # scalar

        # ------------------------------------------------------------------ #
        # HIDDEN LAYER
        # dL/da_hidden — how much each hidden activation affected the loss
        #                = dL/dz_out * the weight connecting it to the output
        # dL/dz_hidden — chain through that hidden neuron's sigmoid
        # dL/dW_hidden — gradient for each hidden weight = dL/dz_h * the input it saw
        # dL/db_hidden — bias gradient = dL/dz_h
        # ------------------------------------------------------------------ #
        dL_da_hidden = dL_dz_out * self.output_neuron.weights  # shape: (n_hidden,)

        for i, neuron in enumerate(self.hidden_neurons):
            dL_dz_h = dL_da_hidden[i] * sigmoid_derivative(neuron.last_z)
            dL_dW_h = dL_dz_h * neuron.last_input
            dL_db_h = dL_dz_h
            neuron.apply_gradients(dL_dW_h, dL_db_h, self.lr)

        self.output_neuron.apply_gradients(dL_dW_out, dL_db_out, self.lr)

    def train(self, inputs, y_true):
        prediction = self.feedforward(inputs)
        loss = binary_cross_entropy(prediction, y_true)
        self.backprop(y_true)
        return loss


if __name__ == "__main__":
    np.random.seed(42)
    network = NeuralNetwork(n_inputs=5, n_hidden=5, learning_rate=0.1)

    # Simple dataset: label=1 when sum of inputs > 15, else 0
    def make_sample():
        x = np.random.uniform(0, 5, size=5)
        y = 1.0 if x.sum() > 15 else 0.0
        return x, y

    print(f"{'Epoch':>6}  {'Loss':>10}  {'Pred':>8}  {'True':>6}")
    print("-" * 38)

    for epoch in range(1, 1001):
        x, y = make_sample()
        loss = network.train(x, y)

        if epoch % 100 == 0:
            pred = network.feedforward(x)
            print(f"{epoch:>6}  {loss:>10.6f}  {pred:>8.4f}  {int(y):>6}")

    # Final accuracy check on 200 fresh samples
    print("\nEvaluating on 200 fresh samples...")
    correct = 0
    for _ in range(200):
        x, y = make_sample()
        pred = network.feedforward(x)
        if round(pred) == y:
            correct += 1
    print(f"Accuracy: {correct}/200 = {correct/2:.1f}%")