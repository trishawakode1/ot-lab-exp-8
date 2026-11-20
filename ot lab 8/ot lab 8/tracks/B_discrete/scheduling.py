"""Track B: makespan scheduling P||Cmax  (assign n jobs to m identical machines, minimise the latest finish time).
Greedy LPT (longest processing time first) vs exact ILP vs brute force."""
import numpy as np, itertools
from scipy.optimize import milp, LinearConstraint, Bounds


def lpt(p, m):
    loads = np.zeros(m); assign = np.zeros(len(p), int)
    for i in np.argsort(-p, kind="stable"): j = int(np.argmin(loads)); assign[i] = j; loads[j] += p[i]
    return float(loads.max()), assign


def ilp_makespan(p, m, time_limit=20):
    n = len(p); nv = n * m + 1                                        # x[i,j] then C
    c = np.zeros(nv); c[-1] = 1
    A_eq = np.zeros((n, nv)); lo_eq = np.ones(n)
    for i in range(n): A_eq[i, i * m:(i + 1) * m] = 1                 # each job on exactly one machine
    A_le = np.zeros((m, nv))
    for j in range(m):
        A_le[j, [i * m + j for i in range(n)]] = p; A_le[j, -1] = -1  # machine load <= C
    ub = np.r_[np.ones(n * m), p.sum()]
    r = milp(c, constraints=[LinearConstraint(A_eq, lo_eq, lo_eq), LinearConstraint(A_le, -np.inf, 0)],
             integrality=np.r_[np.ones(n * m), 0], bounds=Bounds(np.zeros(nv), ub), options={"time_limit": time_limit})
    return float(round(r.fun, 6)), r.status == 0


def brute_force(p, m):
    best = np.inf
    for a in itertools.product(range(m), repeat=len(p) - 1):          # fix job 0 on machine 0 (symmetry)
        loads = np.zeros(m); loads[0] = p[0]
        for i, j in enumerate(a): loads[j] += p[i + 1]
        best = min(best, loads.max())
    return float(best)
