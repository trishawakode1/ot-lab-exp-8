"""Track B: QUBO and Ising formulations, penalty terms, and classical solvers (baselines for Track C).

QUBO:  minimise  E(x) = x^T M x + offset,  x in {0,1}^n,  M symmetric.  (x_i^2 = x_i so M_ii acts as a linear term.)
Ising: minimise  H(s) = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i + const,  s in {-1,+1}^n,  with  x = (1 - s)/2.
"""
import numpy as np


class QUBO:
    def __init__(self, M, offset=0.0): self.M = (M + M.T) / 2; self.offset = offset; self.n = len(M)

    def energy(self, x): x = np.asarray(x, float); return float(x @ self.M @ x + self.offset)

    def energies_all(self, chunk=1 << 16):
        """Energy of every one of the 2^n bitstrings (vectorised). Index k <-> bits of k (bit i = x_i)."""
        out = np.empty(1 << self.n); shifts = np.arange(self.n)
        for s in range(0, 1 << self.n, chunk):
            idx = np.arange(s, min(s + chunk, 1 << self.n)); X = ((idx[:, None] >> shifts) & 1).astype(float)
            out[s:s + len(idx)] = np.einsum("bi,ij,bj->b", X, self.M, X) + self.offset
        return out

    def brute_force(self):
        E = self.energies_all(); k = int(np.argmin(E)); return float(E[k]), ((k >> np.arange(self.n)) & 1).astype(int)

    def to_ising(self):
        """Substitute x=(1-s)/2.  x_i x_j = (1 - s_i - s_j + s_i s_j)/4 ;  x_i = (1 - s_i)/2."""
        M, n = self.M, self.n; J = np.zeros((n, n)); h = np.zeros(n); const = self.offset
        for i in range(n):
            const += M[i, i] / 2; h[i] -= M[i, i] / 2                              # diagonal (linear) part
            for j in range(i + 1, n):
                q = 2 * M[i, j]                                                     # M_ij + M_ji
                const += q / 4; h[i] -= q / 4; h[j] -= q / 4; J[i, j] += q / 4
        return J, h, const

    @staticmethod
    def ising_energy(J, h, const, s): s = np.asarray(s, float); return float(s @ J @ s + h @ s + const)

    def local_search(self, rng, restarts=20):
        """Steepest-descent 1-flip hill climbing from random starts. Fast, gets trapped in local minima."""
        best = (np.inf, None)
        for _ in range(restarts):
            x = rng.integers(0, 2, self.n)
            while True:
                d = (1 - 2 * x) * (np.diag(self.M) + 2 * (self.M @ x - np.diag(self.M) * x))   # energy change of each 1-flip
                i = int(np.argmin(d))
                if d[i] >= -1e-12: break
                x[i] ^= 1
            e = self.energy(x)
            if e < best[0]: best = (e, x.copy())
        return best

    def simulated_annealing(self, rng, sweeps=200, T0=2.0, T1=0.02):
        """Metropolis single-flip annealing, geometric cooling. Evaluations = sweeps * n."""
        x = rng.integers(0, 2, self.n); e = self.energy(x); best = (e, x.copy()); diag = np.diag(self.M)
        for T in np.geomspace(T0, T1, sweeps):
            for i in rng.permutation(self.n):
                d = (1 - 2 * x[i]) * (diag[i] + 2 * (self.M[i] @ x - diag[i] * x[i]))
                if d <= 0 or rng.random() < np.exp(-d / T):
                    x[i] ^= 1; e += d
                    if e < best[0] - 1e-12: best = (e, x.copy())
        return best


def swap_annealing(qb, rng, k, sweeps=100, T0=0.5, T1=0.01):
    """Simulated annealing that PRESERVES sum(x)=k: each move swaps one selected and one unselected variable.
    Needs no penalty term (the constraint is built into the move set), the classical analogue of a constraint-preserving mixer."""
    n = qb.n; x = np.zeros(n, int); x[rng.choice(n, k, replace=False)] = 1; e = qb.energy(x); best = (e, x.copy())
    for T in np.geomspace(T0, T1, sweeps):
        for _ in range(n):
            i = rng.choice(np.flatnonzero(x == 1)); j = rng.choice(np.flatnonzero(x == 0)); y = x.copy(); y[i], y[j] = 0, 1; e2 = qb.energy(y)
            if e2 <= e or rng.random() < np.exp(-(e2 - e) / T):
                x, e = y, e2
                if e < best[0] - 1e-12: best = (e, x.copy())
    return best


def maxcut_qubo(n, edges):
    """Max-Cut: cut(x) = sum_{(i,j)} w (x_i + x_j - 2 x_i x_j).  Minimise -cut  => M_ii = -sum_j w_ij ; M_ij = +w_ij (symmetric, each pair counted twice)."""
    M = np.zeros((n, n))
    for i, j, w in edges: M[i, i] -= w; M[j, j] -= w; M[i, j] += w; M[j, i] += w
    return QUBO(M)


def cardinality_penalty(n, k, P):
    """Penalty P*(sum x - k)^2 = P*( sum_i (1-2k) x_i + 2 sum_{i<j} x_i x_j + k^2 ) using x_i^2=x_i. Zero iff exactly k bits are set."""
    M = np.ones((n, n)) * P; np.fill_diagonal(M, P * (1 - 2 * k)); return QUBO(M, P * k * k)


def add(a, b): return QUBO(a.M + b.M, a.offset + b.offset)
