import numpy as np

# Activations 

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu':    (relu,    relu_derivative),
    'linear':  (lambda z: z, lambda z: np.ones_like(z)),
}

# Layer

class Layer:
    """
    A single fully-connected layer.

    Shapes (batch_size = m):
      input  X  : (m, n_in)
      weights W  : (n_out, n_in)   — one row per output neuron
      bias    b  : (1, n_out)      — broadcast across batch
      output  A  : (m, n_out)
    """

    def __init__(self, n_in, n_out, activation='sigmoid'):
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(ACTIVATIONS)}")

        self.n_in  = n_in
        self.n_out = n_out
        self.act, self.act_d = ACTIVATIONS[activation]

        # Glorot uniform — keeps variance stable regardless of layer size
        limit      = np.sqrt(6 / (n_in + n_out))
        self.W     = np.random.uniform(-limit, limit, (n_out, n_in))
        self.b     = np.zeros((1, n_out))

        # Forward cache — populated during forward(), consumed during backward()
        self._X = None   # input received
        self._Z = None   # pre-activation

        # Gradient accumulators — set during backward(), read by the network
        self.dW = None
        self.db = None

    # Forward pass

    def forward(self, X):
        """
        X : (m, n_in)
        Returns A : (m, n_out)

        Z = X @ W.T + b   →   each row is one sample's pre-activations
        A = act(Z)
        """
        self._X = X
        self._Z = X @ self.W.T + self.b   # (m, n_out)
        return self.act(self._Z)           # (m, n_out)

    # Backward pass

    def backward(self, dL_dA):
        """
        dL_dA : (m, n_out)  — gradient of loss w.r.t. this layer's output,
                               passed down from the layer above (or from the loss).

        Returns dL_dX : (m, n_in)  — gradient to pass to the layer below.

        Chain rule:
          dL/dZ = dL/dA * act'(Z)          element-wise   (m, n_out)
          dL/dW = dL/dZ.T @ X  / m         average over batch   (n_out, n_in)
          dL/db = mean(dL/dZ, axis=0)      average over batch   (1, n_out)
          dL/dX = dL/dZ @ W                pass back to prev layer (m, n_in)
        """
        m = self._X.shape[0]                   # batch size

        dL_dZ = dL_dA * self.act_d(self._Z)    # (m, n_out)

        # Average gradients across the batch before storing
        self.dW = dL_dZ.T @ self._X / m        # (n_out, n_in)
        self.db = dL_dZ.mean(axis=0, keepdims=True)  # (1, n_out)

        return dL_dZ @ self.W                   # (m, n_in)

    # Weight update

    def update(self, lr):
        """Gradient descent step. Called after backward()."""
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def __repr__(self):
        act_name = next(k for k, v in ACTIVATIONS.items() if v[0] is self.act)
        return f"Layer(n_in={self.n_in}, n_out={self.n_out}, activation='{act_name}')"