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
    Fully-connected layer.

    Shapes (m = batch size):
      X  : (m, n_in)
      W  : (n_out, n_in)
      b  : (1,    n_out)
      A  : (m,    n_out)
    """

    def __init__(self, n_in, n_out, activation='sigmoid'):
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. "
                             f"Choose from {list(ACTIVATIONS)}")

        self.n_in  = n_in
        self.n_out = n_out
        self.act, self.act_d = ACTIVATIONS[activation]

        # Glorot uniform initialisation
        limit  = np.sqrt(6 / (n_in + n_out))
        self.W = np.random.uniform(-limit, limit, (n_out, n_in))
        self.b = np.zeros((1, n_out))

        # Forward cache
        self._X = None
        self._Z = None

        # Gradients (written by backward, read by optimiser)
        self.dW = None
        self.db = None

    # Forward

    def forward(self, X):
        self._X = X
        self._Z = X @ self.W.T + self.b
        return self.act(self._Z)

    # Backward

    def backward(self, dL_dA):
        m        = self._X.shape[0]
        dL_dZ    = dL_dA * self.act_d(self._Z)        # (m, n_out)
        self.dW  = dL_dZ.T @ self._X / m              # (n_out, n_in)
        self.db  = dL_dZ.mean(axis=0, keepdims=True)  # (1, n_out)
        return dL_dZ @ self.W                          # (m, n_in)

    # Update

    def update(self, optimizer):
        """
        The layer no longer knows about learning rates or momentum.
        It hands its parameters and gradients to the optimiser,
        which returns the updated values.
        """
        self.W = optimizer.step(self.W, self.dW)
        self.b = optimizer.step(self.b, self.db)

    def __repr__(self):
        act_name = next(k for k, v in ACTIVATIONS.items() if v[0] is self.act)
        return f"Layer(n_in={self.n_in}, n_out={self.n_out}, activation='{act_name}')"