"""Correctness checks for Track A. Run:  python -m unittest test_track_a -v"""
import unittest, numpy as np
from sklearn.datasets import make_moons
import optimizers as O, problems as P

class TestGradients(unittest.TestCase):
    def test_backprop_matches_finite_difference(self):
        X, y = make_moons(60, noise=0.2, random_state=0); m = P.MLP(X, y, hidden=5, l2=0.01)
        th = np.random.default_rng(1).normal(size=m.n_params)
        num = P.numerical_grad(m.loss, th); ana = m.grad(th)
        self.assertLess(np.linalg.norm(num - ana) / np.linalg.norm(num + ana), 1e-6)
    def test_logistic_grad_and_hessian(self):
        X, y = make_moons(50, noise=0.2, random_state=1); m = P.Logistic(X, y, 0.1)
        th = np.array([0.3, -0.2, 0.1])
        self.assertLess(np.linalg.norm(P.numerical_grad(m.loss, th) - m.grad(th)), 1e-6)
        Hn = np.array([P.numerical_grad(lambda t: m.grad(t)[i], th) for i in range(3)])
        self.assertLess(np.linalg.norm(Hn - m.hess(th)), 1e-5)
    def test_rosenbrock_and_saddle_grads(self):
        f, g, h = P.rosenbrock(); x = np.array([0.7, -0.4])
        self.assertLess(np.linalg.norm(P.numerical_grad(f, x) - g(x)), 1e-4)
        f, g, h = P.saddle()
        self.assertLess(np.linalg.norm(P.numerical_grad(f, x) - g(x)), 1e-6)
        f, g, h = P.double_well(); x1 = np.array([0.7])
        self.assertLess(np.linalg.norm(P.numerical_grad(f, x1) - g(x1)), 1e-6)
    def test_hvp_matches_exact_hessian(self):
        f, g, h = P.rosenbrock(); x = np.array([0.3, 0.2]); v = np.array([1.0, -2.0])
        self.assertLess(np.linalg.norm(O.hvp_fd(g, x, v) - h(x) @ v), 1e-4)

class TestOptimizers(unittest.TestCase):
    def test_all_first_order_solve_easy_quadratic(self):
        f, g, _, _ = P.quadratic(kappa=4, d=5)
        for name, lr in [("GD", .3), ("Momentum", .05), ("NAG", .05), ("AdaGrad", .5), ("RMSProp", .01), ("Adam", .05)]:
            x, fs, _ = O.minimize(O.FIRST_ORDER[name](lr), f, g, np.ones(5), 3000)
            self.assertLess(fs[-1], 1e-3, name)
    def test_newton_one_step_on_quadratic(self):
        f, g, h, _ = P.quadratic(kappa=1000, d=6); x, fs = O.newton(f, g, h, np.ones(6), steps=1)
        self.assertLess(np.linalg.norm(x), 1e-9)               # exact in ONE step regardless of conditioning
    def test_newton_cg_matches_newton(self):
        f, g, h = P.rosenbrock(); x1, _ = O.newton(f, g, h, np.array([-1.2, 1.0]), 60, line_search=True)
        x2, _ = O.newton_cg(f, g, np.array([-1.2, 1.0]), 60)
        self.assertLess(np.linalg.norm(x1 - np.array([1, 1])), 1e-4); self.assertLess(np.linalg.norm(x2 - np.array([1, 1])), 1e-3)
    def test_gauss_newton_recovers_exponential_params(self):
        t = np.linspace(0, 2, 30); true = np.array([2.0, -1.3]); y = true[0] * np.exp(true[1] * t)
        r = lambda p: p[0] * np.exp(p[1] * t) - y
        J = lambda p: np.stack([np.exp(p[1] * t), p[0] * t * np.exp(p[1] * t)], 1)
        x, _ = O.gauss_newton(r, J, np.array([1.0, -0.5]), 30)
        self.assertLess(np.linalg.norm(x - true), 1e-8)
    def test_gd_rate_matches_theory(self):
        kappa = 20; f, g, _, A = P.quadratic(kappa, d=8); lr = 2 / (1 + kappa)   # optimal fixed step, mu=1, L=kappa
        x0 = np.ones(8); _, fs, _ = O.minimize(O.GD(lr), f, g, x0, 200)
        emp = (fs[100] / fs[50]) ** (1 / 50); theo = ((kappa - 1) / (kappa + 1)) ** 2
        self.assertLess(abs(emp - theo), 0.02)

if __name__ == "__main__": unittest.main()
