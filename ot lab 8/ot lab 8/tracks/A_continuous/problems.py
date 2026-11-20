"""Track A objective functions and ML models (numpy only)."""
import numpy as np

# ------------------------------------------------------------------ analytic test functions ----
def quadratic(kappa, d=10, seed=0):
    """Strongly convex f = 0.5 x^T A x, eigenvalues spaced in [1, kappa]. Minimum 0 at origin."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = Q @ np.diag(np.linspace(1, kappa, d)) @ Q.T
    return (lambda x: 0.5 * x @ A @ x), (lambda x: A @ x), (lambda x: A), A

def rosenbrock():
    f = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = lambda x: np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])
    h = lambda x: np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200.0]])
    return f, g, h                                   # non-convex, global min 0 at (1,1)

def saddle():
    """f = x^2 - y^2 + 0.25 y^4 : saddle at origin, two global minima at (0, +-sqrt2) with f=-1."""
    f = lambda p: p[0]**2 - p[1]**2 + 0.25 * p[1]**4
    g = lambda p: np.array([2 * p[0], -2 * p[1] + p[1]**3])
    h = lambda p: np.array([[2.0, 0.0], [0.0, -2 + 3 * p[1]**2]])
    return f, g, h

def double_well():
    """1-D non-convex: f = x^4 - 3x^2 + x. Local min near x=+1.13 (f~-1.07), global near x=-1.30 (f~-3.51)."""
    f = lambda x: x[0]**4 - 3 * x[0]**2 + x[0]
    g = lambda x: np.array([4 * x[0]**3 - 6 * x[0] + 1])
    h = lambda x: np.array([[12 * x[0]**2 - 6]])
    return f, g, h

def convex_1d():
    f = lambda x: x[0]**4                           # convex but NOT strongly convex (flat minimum)
    g = lambda x: np.array([4 * x[0]**3]); h = lambda x: np.array([[12 * x[0]**2]])
    return f, g, h

# ------------------------------------------------------------------------ logistic regression ----
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class Logistic:
    """Regularised logistic regression: loss = mean CE + (l2/2)||w||^2 (bias unregularised).
    theta = [w, b]. Convex; strongly convex when l2>0."""
    def __init__(self, X, y, l2=1e-3): self.X, self.y, self.l2 = X, y, l2
    def _split(self, th): return th[:-1], th[-1]
    def loss(self, th, idx=None):
        X, y = (self.X, self.y) if idx is None else (self.X[idx], self.y[idx])
        w, b = self._split(th); p = sigmoid(X @ w + b); e = 1e-12
        return -np.mean(y * np.log(p + e) + (1 - y) * np.log(1 - p + e)) + 0.5 * self.l2 * w @ w
    def grad(self, th, idx=None):
        X, y = (self.X, self.y) if idx is None else (self.X[idx], self.y[idx])
        w, b = self._split(th); r = sigmoid(X @ w + b) - y
        return np.concatenate([X.T @ r / len(y) + self.l2 * w, [r.mean()]])
    def hess(self, th):
        w, b = self._split(th); p = sigmoid(self.X @ w + b); s = p * (1 - p)
        Xa = np.hstack([self.X, np.ones((len(self.y), 1))])
        H = (Xa * s[:, None]).T @ Xa / len(self.y); H[:-1, :-1] += self.l2 * np.eye(len(w)); return H
    def per_sample_grads(self, th):
        w, b = self._split(th); r = sigmoid(self.X @ w + b) - self.y
        return np.hstack([self.X * r[:, None], r[:, None]])
    def accuracy(self, th): w, b = self._split(th); return np.mean((self.X @ w + b > 0) == (self.y == 1))

# --------------------------------------------------------------- tiny MLP with manual backprop ----
class MLP:
    """2-layer tanh MLP for binary classification, with HAND-WRITTEN backpropagation.
    Parameters are packed in one flat vector so any optimizer can be applied. Non-convex loss."""
    def __init__(self, X, y, hidden=8, l2=0.0, seed=0):
        self.X, self.y, self.h, self.l2 = X, y, hidden, l2
        self.d = X.shape[1]; rng = np.random.default_rng(seed)
        self.n_params = self.d * hidden + hidden + hidden + 1
        self.theta0 = np.concatenate([rng.normal(0, 0.5, self.d * hidden), np.zeros(hidden),
                                      rng.normal(0, 0.5, hidden), [0.0]])
    def unpack(self, th):
        d, h = self.d, self.h; i = 0
        W1 = th[i:i + d * h].reshape(d, h); i += d * h
        b1 = th[i:i + h]; i += h
        W2 = th[i:i + h]; i += h
        return W1, b1, W2, th[i]
    def forward(self, th, X):
        W1, b1, W2, b2 = self.unpack(th)
        A1 = np.tanh(X @ W1 + b1); z = A1 @ W2 + b2
        return A1, z
    def loss(self, th, idx=None):
        X, y = (self.X, self.y) if idx is None else (self.X[idx], self.y[idx])
        _, z = self.forward(th, X)
        ce = np.mean(np.logaddexp(0, z) - y * z)                     # stable BCE-with-logits
        return ce + 0.5 * self.l2 * (th @ th)
    def grad(self, th, idx=None):
        """Backpropagation: chain rule applied layer by layer, from the loss back to W1."""
        X, y = (self.X, self.y) if idx is None else (self.X[idx], self.y[idx])
        W1, b1, W2, b2 = self.unpack(th); n = len(y)
        A1, z = self.forward(th, X)
        dz = (sigmoid(z) - y) / n                                    # dL/dz
        dW2 = A1.T @ dz; db2 = dz.sum()                              # layer 2
        dA1 = np.outer(dz, W2)                                       # propagate to hidden activations
        dZ1 = dA1 * (1 - A1**2)                                      # tanh' = 1 - tanh^2
        dW1 = X.T @ dZ1; db1 = dZ1.sum(0)                            # layer 1
        return np.concatenate([dW1.ravel(), db1, dW2, [db2]]) + self.l2 * th
    def accuracy(self, th): _, z = self.forward(th, self.X); return np.mean((z > 0) == (self.y == 1))

def numerical_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = eps; g[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return g
