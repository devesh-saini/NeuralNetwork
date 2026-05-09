import numpy as np
from Layer import Layer
from Optimizers import SGD, RMSprop, Adam

# Loss

def binary_cross_entropy(A, y):
    A    = np.clip(A, 1e-12, 1 - 1e-12)
    loss = -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))
    grad = (-(y / A) + (1 - y) / (1 - A)) / A.shape[0]
    return loss, grad

# Network

class NeuralNetwork:
    def __init__(self, layers: list[Layer], optimizer):
        self.layers    = layers
        self.optimizer = optimizer

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self):
        for layer in self.layers:
            layer.update(self.optimizer)

    def train(self, X, y, epochs=500, batch_size=32, print_every=100):
        m = X.shape[0]
        history = []

        print(f"Optimizer : {self.optimizer}")
        print(f"Samples   : {m}  |  Epochs: {epochs}  |  Batch: {batch_size}\n")

        for epoch in range(1, epochs + 1):
            idx  = np.random.permutation(m)
            X_s, y_s = X[idx], y[idx]

            epoch_loss = 0
            n_batches  = 0

            for start in range(0, m, batch_size):
                Xb = X_s[start:start + batch_size]
                yb = y_s[start:start + batch_size]

                A              = self.forward(Xb)
                loss, dL_dA    = binary_cross_entropy(A, yb)
                epoch_loss    += loss
                n_batches     += 1

                self.backward(dL_dA)
                self.update()

            avg = epoch_loss / n_batches
            history.append(avg)

            if epoch % print_every == 0 or epoch == 1:
                print(f"  epoch {epoch:>5}  loss={avg:.6f}")

        print()
        return history

    def predict(self, X):
        return self.forward(X)

    def __repr__(self):
        lines = ["NeuralNetwork("]
        for i, l in enumerate(self.layers):
            lines.append(f"  [{i}] {l}")
        lines.append(f"  opt={self.optimizer}")
        lines.append(")")
        return "\n".join(lines)


# Demo

if __name__ == "__main__":
    np.random.seed(42)

    def make_data(n, n_features=5):
        X = np.random.randn(n, n_features)
        y = (X.sum(axis=1) > 0).astype(float).reshape(-1, 1)
        return X, y

    X_train, y_train = make_data(500)
    X_test,  y_test  = make_data(100)

    def make_layers():
        return [
            Layer(5, 16, activation='relu'),
            Layer(16, 8, activation='relu'),
            Layer(8,  1, activation='sigmoid'),
        ]

    configs = [
        ("SGD + Momentum", SGD(lr=0.05,  momentum=0.9)),
        ("RMSprop",        RMSprop(lr=0.001)),
        ("Adam",           Adam(lr=0.001)),
    ]

    results = {}

    for name, opt in configs:
        print(f"{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        net     = NeuralNetwork(make_layers(), optimizer=opt)
        history = net.train(X_train, y_train, epochs=300, batch_size=32, print_every=100)
        preds   = (net.predict(X_test) > 0.5).astype(float)
        acc     = (preds == y_test).mean()
        results[name] = (history[0], history[-1], acc)
        print(f"  Accuracy : {acc*100:.1f}%\n")

    print(f"{'─'*50}")
    print(f"  {'Optimiser':<18} {'Loss start':>10} {'Loss end':>10} {'Accuracy':>10}")
    print(f"{'─'*50}")
    for name, (start, end, acc) in results.items():
        print(f"  {name:<18} {start:>10.6f} {end:>10.6f} {acc*100:>9.1f}%")
    print(f"{'─'*50}")