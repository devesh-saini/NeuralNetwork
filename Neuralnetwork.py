import numpy as np
from Layer import Layer

# ── Loss functions ─────────────────────────────────────────────────────────────

def binary_cross_entropy(A, y):
    """
    A : (m, 1)  predictions
    y : (m, 1)  true labels
    Returns scalar mean loss and gradient dL/dA : (m, 1)
    """
    A   = np.clip(A, 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))
    grad = -(y / A) + (1 - y) / (1 - A)          # dL/dA, shape (m,1)
    return loss, grad / A.shape[0]                # normalise by batch size

# ── Network ───────────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    A stack of Layer objects.
    Build any architecture by passing a list of Layer instances.

    Example:
        net = NeuralNetwork([
            Layer(5, 8, activation='relu'),
            Layer(8, 4, activation='relu'),
            Layer(4, 1, activation='sigmoid'),
        ])
    """

    def __init__(self, layers: list[Layer]):
        self.layers = layers

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, X):
        """Pass X through every layer in sequence."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A                   # final output, shape (m, n_out_last)

    # ── Backward ──────────────────────────────────────────────────────────────

    def backward(self, grad):
        """
        Propagate the loss gradient back through layers in reverse.
        Each layer's backward() returns the gradient for the layer below it.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, X, y, epochs=1000, lr=0.1, batch_size=32, print_every=100):
        """
        Mini-batch gradient descent.

        Each epoch:
          1. Shuffle the dataset
          2. Slice into batches of size batch_size
          3. Forward → loss → backward → update for each batch
          4. Track mean epoch loss for diagnostics
        """
        m = X.shape[0]
        loss_history = []

        print(f"Training  layers={[str(l) for l in self.layers]}")
        print(f"          samples={m}  epochs={epochs}  lr={lr}  batch={batch_size}\n")

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(m)
            X_s, y_s = X[idx], y[idx]

            epoch_loss = 0
            n_batches  = 0

            for start in range(0, m, batch_size):
                X_batch = X_s[start : start + batch_size]   # (batch, n_in)
                y_batch = y_s[start : start + batch_size]   # (batch, 1)

                # Forward
                A = self.forward(X_batch)

                # Loss
                loss, dL_dA = binary_cross_entropy(A, y_batch)
                epoch_loss += loss
                n_batches  += 1

                # Backward
                self.backward(dL_dA)

                # Update weights
                self.update(lr)

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if epoch % print_every == 0 or epoch == 1:
                print(f"  epoch {epoch:>5}  loss={avg_loss:.6f}")

        print()
        return loss_history

    def predict(self, X):
        return self.forward(X)

    def __repr__(self):
        lines = ["NeuralNetwork("]
        for i, l in enumerate(self.layers):
            lines.append(f"  [{i}] {l}")
        lines.append(")")
        return "\n".join(lines)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # Same task as before: predict whether sum of inputs > 0
    def make_data(n, n_features=5):
        X = np.random.randn(n, n_features)
        y = (X.sum(axis=1) > 0).astype(float).reshape(-1, 1)
        return X, y

    X_train, y_train = make_data(500)
    X_test,  y_test  = make_data(100)

    net = NeuralNetwork([
        Layer(5, 8,  activation='relu'),
        Layer(8, 4,  activation='relu'),
        Layer(4, 1,  activation='sigmoid'),
    ])

    print(net)
    print()

    history = net.train(X_train, y_train, epochs=500, lr=0.05, batch_size=32, print_every=100)

    preds  = net.predict(X_test)
    labels = (preds > 0.5).astype(float)
    acc    = (labels == y_test).mean()

    print(f"Test accuracy : {acc * 100:.1f}%")
    print(f"Loss start    : {history[0]:.6f}")
    print(f"Loss end      : {history[-1]:.6f}")