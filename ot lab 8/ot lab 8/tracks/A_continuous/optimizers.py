"""Track A: first-order and second-order optimizers, implemented from scratch (numpy only).

Every first-order optimizer exposes  step(x, grad_fn) -> new_x  so that Nesterov (which needs
the gradient at a look-ahead point) and the others share one interface.
"""
import numpy as np


class GD:
    name = "GD"
    def __init__(self, lr): self.lr = lr
    def step(self, x, grad_fn): return x - self.lr * grad_fn(x)


class Momentum:
    """Heavy-ball: v <- beta v + g ;  x <- x - lr v."""
    name = "Momentum"
    def __init__(self, lr, beta=0.9): self.lr, self.beta, self.v = lr, beta, None
    def step(self, x, grad_fn):
        g = grad_fn(x)
        self.v = g if self.v is None else self.beta * self.v + g
        return x - self.lr * self.v


class NAG:
    """Nesterov: gradient evaluated at the look-ahead point x + beta*v."""
    name = "NAG"
    def __init__(self, lr, beta=0.9): self.lr, self.beta, self.v = lr, beta, None
    def step(self, x, grad_fn):
        if self.v is None: self.v = np.zeros_like(x)
        g = grad_fn(x + self.beta * self.v)
        self.v = self.beta * self.v - self.lr * g
        return x + self.v


class AdaGrad:
    name = "AdaGrad"
    def __init__(self, lr, eps=1e-8): self.lr, self.eps, self.G = lr, eps, None
    def step(self, x, grad_fn):
        g = grad_fn(x)
        self.G = g**2 if self.G is None else self.G + g**2
        return x - self.lr * g / (np.sqrt(self.G) + self.eps)


class RMSProp:
    name = "RMSProp"
    def __init__(self, lr, rho=0.9, eps=1e-8): self.lr, self.rho, self.eps, self.s = lr, rho, eps, None
    def step(self, x, grad_fn):
        g = grad_fn(x)
        self.s = (1 - self.rho) * g**2 if self.s is None else self.rho * self.s + (1 - self.rho) * g**2
        return x - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    name = "Adam"
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None; self.t = 0
    def step(self, x, grad_fn):
        g = grad_fn(x); self.t += 1
        if self.m is None: self.m, self.v = np.zeros_like(x), np.zeros_like(x)
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g**2
        mh, vh = self.m / (1 - self.b1**self.t), self.v / (1 - self.b2**self.t)   # bias correction
        return x - self.lr * mh / (np.sqrt(vh) + self.eps)


FIRST_ORDER = {"GD": GD, "Momentum": Momentum, "NAG": NAG, "AdaGrad": AdaGrad, "RMSProp": RMSProp, "Adam": Adam}


def minimize(opt, f, grad, x0, steps):
    """Run `steps` iterations. Returns (x_final, f_history, grad_norm_history). Divergence -> inf."""
    x = np.array(x0, float); fs = [f(x)]; gn = [np.linalg.norm(grad(x))]
    for _ in range(steps):
        x = opt.step(x, grad)
        fx = f(x)
        if not np.isfinite(fx) or np.linalg.norm(x) > 1e12:
            fs.append(np.inf); gn.append(np.inf); break
        fs.append(fx); gn.append(np.linalg.norm(grad(x)))
    return x, np.array(fs), np.array(gn)


# ---------------------------------------------------------------- second-order / curvature ----
def newton(f, grad, hess, x0, steps=50, damping=0.0, line_search=False):
    """Newton: x <- x - (H + damping I)^-1 g.  NOTE: converges to ANY stationary point
    (including saddles) because it is attracted to g=0, not to minima."""
    x = np.array(x0, float); fs = [f(x)]
    for _ in range(steps):
        g = grad(x)
        if np.linalg.norm(g) < 1e-12: break
        d = np.linalg.solve(hess(x) + damping * np.eye(len(x)), g)
        t = 1.0
        if line_search:                      # backtracking Armijo
            while f(x - t * d) > f(x) - 1e-4 * t * g @ d and t > 1e-10: t *= 0.5
        x = x - t * d; fs.append(f(x))
    return x, np.array(fs)


def gauss_newton(residual, jac, x0, steps=50, lam=0.0):
    """Gauss-Newton for  min 0.5||r(x)||^2 :  solve (J^T J + lam I) d = J^T r ;  x <- x - d.
    lam>0 gives Levenberg-Marquardt damping. Drops the second-derivative term of the Hessian."""
    x = np.array(x0, float); fs = [0.5 * residual(x) @ residual(x)]
    for _ in range(steps):
        r, J = residual(x), jac(x)
        A = J.T @ J + lam * np.eye(len(x))
        d = np.linalg.solve(A, J.T @ r)
        x = x - d; fs.append(0.5 * residual(x) @ residual(x))
        if not np.isfinite(fs[-1]): break
    return x, np.array(fs)


def hvp_fd(grad, x, v, eps=1e-5):
    """Hessian-vector product by central finite difference of gradients: H v ~ (g(x+eps v)-g(x-eps v))/(2 eps).
    Costs two gradient evaluations; never forms the d x d Hessian."""
    return (grad(x + eps * v) - grad(x - eps * v)) / (2 * eps)


def cg_solve(matvec, b, iters=100, tol=1e-10):
    """Conjugate gradient for  A x = b  using only matrix-vector products."""
    x = np.zeros_like(b); r = b.copy(); p = r.copy(); rs = r @ r
    for _ in range(iters):
        if rs < tol**2: break
        Ap = matvec(p); a = rs / (p @ Ap)
        x += a * p; r -= a * Ap; rs_new = r @ r
        p = r + (rs_new / rs) * p; rs = rs_new
    return x


def newton_cg(f, grad, x0, steps=30, damping=1e-6):
    """Hessian-free Newton: the Newton system is solved by CG with Hessian-vector products."""
    x = np.array(x0, float); fs = [f(x)]
    for _ in range(steps):
        g = grad(x)
        if np.linalg.norm(g) < 1e-10: break
        d = cg_solve(lambda v: hvp_fd(grad, x, v) + damping * v, g, iters=len(x) * 2)
        t = 1.0
        while f(x - t * d) > f(x) - 1e-4 * t * g @ d and t > 1e-10: t *= 0.5
        x = x - t * d; fs.append(f(x))
    return x, np.array(fs)
