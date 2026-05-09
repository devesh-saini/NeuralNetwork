import numpy as np
from Neuron import Neuron, sigmoid, sigmoid_derivative

def random_weights(n_in):
    """Glorot uniform — keeps activation variance stable across layers."""
    limit = np.sqrt(6 / (n_in + 1))
    return np.random.uniform(-limit, limit, size=n_in)

# Loss

def binary_cross_entropy(y_pred, y_true):
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_derivative(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

# Network

class NeuralNetwork:
    def __init__(self, n_inputs=5, n_hidden=5):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        self.hidden_neurons = [
            Neuron(random_weights(n_inputs), bias=np.random.randn() * 0.1)
            for _ in range(n_hidden)
        ]
        self.output_neuron = Neuron(random_weights(n_hidden), bias=np.random.randn() * 0.1)

        # Cached during forward pass
        self.hidden_outputs = None

    def feedforward(self, inputs):
        self.hidden_outputs = np.array([
            n.feedforward(inputs) for n in self.hidden_neurons
        ])
        return self.output_neuron.feedforward(self.hidden_outputs)

    def backprop(self, y_pred, y_true, lr=0.01):
        """
        Chain rule, output → hidden.

        Notation:
          z   = pre-activation (dot product + bias)
          a   = post-activation (sigmoid(z))
          L   = loss
        """

        # Output layer
        # dL/da_out  — how loss changes with the output neuron's activation
        dL_da_out = bce_derivative(y_pred, y_true)

        # da_out/dz_out  — how the activation changes with its pre-activation
        da_dz_out = sigmoid_derivative(self.output_neuron.last_z)

        # Chain: dL/dz_out = dL/da_out * da_out/dz_out
        dL_dz_out = dL_da_out * da_dz_out

        # dL/dW_out = dL/dz_out * dz_out/dW  = dL/dz_out * hidden_outputs
        dL_dW_out = dL_dz_out * self.hidden_outputs  # shape: (n_hidden,)
        dL_db_out = dL_dz_out                         # bias gradient is just dL/dz

        # Hidden layer
        # Propagate the error signal back through the output weights.
        # dL/da_hidden[i] = dL/dz_out * W_out[i]
        # (Each hidden neuron's activation affects the output via its weight)
        dL_da_hidden = dL_dz_out * self.output_neuron.weights  # shape: (n_hidden,)

        dL_dW_hidden = []
        dL_db_hidden = []

        for i, neuron in enumerate(self.hidden_neurons):
            # da_hidden/dz_hidden — sigmoid derivative at this neuron's pre-activation
            da_dz_h = sigmoid_derivative(neuron.last_z)

            # dL/dz_hidden[i] = dL/da_hidden[i] * da_hidden/dz_hidden
            dL_dz_h = dL_da_hidden[i] * da_dz_h

            # dL/dW_hidden[i] = dL/dz_h * input
            dL_dW_h = dL_dz_h * neuron.last_input
            dL_db_h = dL_dz_h

            dL_dW_hidden.append(dL_dW_h)
            dL_db_hidden.append(dL_db_h)

        # Apply updates
        self.output_neuron.update(dL_dW_out, dL_db_out, lr)
        for i, neuron in enumerate(self.hidden_neurons):
            neuron.update(dL_dW_hidden[i], dL_db_hidden[i], lr)

    def train(self, X, y, epochs=1000, lr=0.1, print_every=100):
        print(f"Training: {len(X)} samples, {epochs} epochs, lr={lr}\n")
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for xi, yi in zip(X, y):
                pred = self.feedforward(xi)
                total_loss += binary_cross_entropy(pred, yi)
                self.backprop(pred, yi, lr)

            if epoch % print_every == 0 or epoch == 1:
                avg_loss = total_loss / len(X)
                print(f"  epoch {epoch:>5}  loss={avg_loss:.6f}")
        print()

    def predict(self, X):
        return np.array([self.feedforward(xi) for xi in X])


if __name__ == "__main__":
    np.random.seed(42)

    # XOR-style problem extended to 5 inputs.
    # Label = 1 if more than half the inputs are positive, else 0.
    def make_data(n_samples=200, n_features=5):
        X = np.random.randn(n_samples, n_features)
        y = (X.sum(axis=1) > 0).astype(float)
        return X, y

    X_train, y_train = make_data(200)
    X_test,  y_test  = make_data(50)

    net = NeuralNetwork(n_inputs=5, n_hidden=5)
    net.train(X_train, y_train, epochs=1000, lr=0.1, print_every=200)

    # Evaluate
    preds = net.predict(X_test)
    labels = (preds > 0.5).astype(float)
    acc = (labels == y_test).mean()

    print(f"Test accuracy : {acc*100:.1f}%")
    print(f"Sample preds  : {preds[:5].round(3)}")
    print(f"True labels   : {y_test[:5].astype(int)}")