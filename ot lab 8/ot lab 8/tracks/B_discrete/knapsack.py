"""Track B: 0/1 knapsack solved four ways: greedy, dynamic programming, Branch-and-Bound (own code), ILP (scipy/HiGHS).
Knapsack is the template for budgeted selection: choose items (features, indexes, tests) maximising value under a budget."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def greedy(v, w, cap, best_single=False):
    """Take items by value/weight ratio while they fit. No optimality guarantee; with best_single=True
    (compare with the single most valuable item) it becomes the classic 1/2-approximation."""
    order = np.argsort(-v / w, kind="stable"); mask = np.zeros(len(v), bool); room = cap
    for i in order:
        if w[i] <= room: mask[i] = True; room -= w[i]
    if best_single:
        fits = np.where(w <= cap)[0]
        if len(fits):
            j = fits[np.argmax(v[fits])]
            if v[j] > v[mask].sum(): mask = np.zeros(len(v), bool); mask[j] = True
    return int(v[mask].sum()), mask


def dp(v, w, cap):
    """Exact dynamic programming, O(n*cap) time and memory (pseudo-polynomial). Backtracks the chosen set."""
    n = len(v); best = np.zeros(cap + 1, np.int64); keep = np.zeros((n, cap + 1), bool)
    for i in range(n):
        if w[i] > cap: continue
        cand = best[:cap + 1 - w[i]] + v[i]                      # value if item i is added to capacity c-w[i]
        better = cand > best[w[i]:]
        keep[i, w[i]:] = better
        best[w[i]:] = np.where(better, cand, best[w[i]:])
    mask = np.zeros(n, bool); c = cap
    for i in range(n - 1, -1, -1):
        if keep[i, c]: mask[i] = True; c -= w[i]
    return int(best[cap]), mask


def branch_and_bound(v, w, cap, node_limit=3_000_000, time_limit=5.0):
    """Depth-first Branch-and-Bound. Items sorted by ratio; bound = value of the fractional (LP) relaxation of what is left.
    A node is pruned when floor(bound) cannot beat the incumbent. Stops at node_limit or time_limit seconds.
    Returns (value, mask, nodes, finished) where finished=True means optimality was PROVEN."""
    import time; t0 = time.perf_counter(); stop = [False]
    n = len(v); order = np.argsort(-v / w, kind="stable"); vs, ws = v[order].astype(float), w[order].astype(float)
    best = [0, np.zeros(n, bool)]; nodes = [0]; cur = np.zeros(n, bool)

    def bound(i, room, val):
        b = val
        while i < n and ws[i] <= room: room -= ws[i]; b += vs[i]; i += 1
        if i < n: b += vs[i] * room / ws[i]                      # fractional part
        return b

    def rec(i, room, val):
        nodes[0] += 1
        if stop[0]: return
        if nodes[0] > node_limit or (nodes[0] & 1023 == 0 and time.perf_counter() - t0 > time_limit): stop[0] = True; return
        if val > best[0]: best[0] = val; best[1] = cur.copy()
        if i == n or int(bound(i, room, val)) <= best[0]: return  # prune
        if ws[i] <= room: cur[i] = True; rec(i + 1, room - ws[i], val + vs[i]); cur[i] = False   # branch: take
        rec(i + 1, room, val)                                                                      # branch: skip
    import sys; sys.setrecursionlimit(10000); rec(0, float(cap), 0.0)
    mask = np.zeros(n, bool); mask[order] = best[1]
    return int(round(best[0])), mask, nodes[0], not stop[0]


def ilp(v, w, cap, time_limit=30):
    """Integer linear program  max v.x  s.t. w.x <= cap, x in {0,1}^n, solved by HiGHS (branch-and-cut)."""
    r = milp(-v.astype(float), constraints=LinearConstraint(w[None, :].astype(float), -np.inf, cap),
             integrality=np.ones(len(v)), bounds=Bounds(0, 1), options={"time_limit": time_limit})
    mask = np.round(r.x).astype(bool) if r.x is not None else np.zeros(len(v), bool)
    return int(v[mask].sum()), mask, r.status == 0


def brute_force(v, w, cap):
    n = len(v); best = 0; bm = 0
    for m in range(1 << n):
        bits = (m >> np.arange(n)) & 1
        if w @ bits <= cap and v @ bits > best: best, bm = int(v @ bits), m
    return best, ((bm >> np.arange(n)) & 1).astype(bool)


def make_instance(n, kind, seed):
    rng = np.random.default_rng(seed); w = rng.integers(10, 101, n)
    v = rng.integers(10, 101, n) if kind == "uncorrelated" else w + 10     # strongly correlated = hard for B&B
    return v.astype(np.int64), w.astype(np.int64), int(w.sum() // 2)
