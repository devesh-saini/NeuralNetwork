import numpy as np

# Base

class Optimizer:
    """
    Every optimiser implements one method:
        step(param, grad) -> updated param

    State (momentum buffers, etc.) is stored on the optimiser instance
    keyed by the id() of the parameter array so each Layer's W and b
    get their own independent accumulators.
    """
    def step(self, param, grad):
        raise NotImplementedError

# SGD + Momentum

class SGD(Optimizer):
    """
    v_t = beta * v_{t-1} - lr * grad
    W   = W + v_t

    beta=0.0  →  vanilla SGD (no momentum)
    beta=0.9  →  typical momentum value
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr       = lr
        self.momentum = momentum
        self._v       = {}          # velocity per parameter

    def step(self, param, grad):
        pid = (id(param), param.shape)
        if pid not in self._v:
            self._v[pid] = np.zeros_like(param)

        self._v[pid] = self.momentum * self._v[pid] - self.lr * grad
        return param + self._v[pid]

    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum})"

# RMSprop

class RMSprop(Optimizer):
    """
    s_t = beta * s_{t-1} + (1 - beta) * grad^2
    W   = W - lr * grad / (sqrt(s_t) + eps)

    Dividing by the root of the running squared gradient normalises the
    step size per weight — large-gradient weights get smaller steps.
    """
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr   = lr
        self.beta = beta
        self.eps  = eps
        self._s   = {}             # second moment per parameter

    def step(self, param, grad):
        pid = (id(param), param.shape)
        if pid not in self._s:
            self._s[pid] = np.zeros_like(param)

        self._s[pid] = self.beta * self._s[pid] + (1 - self.beta) * grad ** 2
        return param - self.lr * grad / (np.sqrt(self._s[pid]) + self.eps)

    def __repr__(self):
        return f"RMSprop(lr={self.lr}, beta={self.beta})"

# Adam

class Adam(Optimizer):
    """
    m_t = b1 * m_{t-1} + (1 - b1) * grad          ← first moment  (mean)
    v_t = b2 * v_{t-1} + (1 - b2) * grad^2        ← second moment (variance)

    Bias correction (critical in early steps when m and v are near zero):
      m_hat = m_t / (1 - b1^t)
      v_hat = v_t / (1 - b2^t)

    W = W - lr * m_hat / (sqrt(v_hat) + eps)

    Default hyperparameters are from the original Adam paper (Kingma & Ba 2014)
    and work well across most problems without tuning.
    """
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr  = lr
        self.b1  = b1
        self.b2  = b2
        self.eps = eps
        self._m  = {}              # first moment per parameter
        self._v  = {}              # second moment per parameter
        self._t  = {}              # step count per parameter

    def step(self, param, grad):
        pid = (id(param), param.shape)
        if pid not in self._m:
            self._m[pid] = np.zeros_like(param)
            self._v[pid] = np.zeros_like(param)
            self._t[pid] = 0

        self._t[pid] += 1
        t = self._t[pid]

        self._m[pid] = self.b1 * self._m[pid] + (1 - self.b1) * grad
        self._v[pid] = self.b2 * self._v[pid] + (1 - self.b2) * grad ** 2

        # Bias correction — removes the initialisation bias toward zero
        m_hat = self._m[pid] / (1 - self.b1 ** t)
        v_hat = self._v[pid] / (1 - self.b2 ** t)

        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self):
        return f"Adam(lr={self.lr}, b1={self.b1}, b2={self.b2})"