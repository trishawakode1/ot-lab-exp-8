"""Track A experiment runner. Produces results/*.csv and figures/*.png. Deterministic (fixed seeds)."""
import time, csv, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.preprocessing import StandardScaler
import optimizers as O, problems as P

RES, FIG = "results/", "figures/"
LRS = [10.0**k for k in np.arange(-5, 2.01, 0.5)]   # 1e-5 .. 1e2
OPTS = list(O.FIRST_ORDER)
def median_time(fn, reps=7):
    ts = []
    for _ in range(reps): t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
def savefig(n): plt.tight_layout(); plt.savefig(FIG + n, dpi=130); plt.close()

def tune(name, f, g, x0, steps):
    """Pick lr minimising final loss at a FIXED iteration budget. Same grid for every optimizer.
    Returns best lr, its history, and the full lr->final-loss curve (the sensitivity curve)."""
    curve, best = {}, (np.inf, None, None)
    for lr in LRS:
        _, fs, _ = O.minimize(O.FIRST_ORDER[name](lr), f, g, x0, steps)
        curve[lr] = fs[-1]
        if fs[-1] < best[0]: best = (fs[-1], lr, fs)
    return best[1], best[2], curve

# ====================================================== 1. landscapes: minima, saddle, convexity ====
def landscapes():
    out = {}
    f, g, h = P.double_well(); fin = {}
    for x0 in [-2.0, -0.2, 0.3, 2.0]:
        x, fs, _ = O.minimize(O.GD(0.01), f, g, [x0], 2000); fin[x0] = (float(x[0]), float(fs[-1]))
    out["double_well_GD_from_start"] = fin                      # different starts -> different minima
    f, g, h = P.saddle(); r = {}
    x, fs, _ = O.minimize(O.GD(0.1), f, g, [1.0, 0.0], 500); r["GD_exact_axis_start(1,0)"] = (x.tolist(), float(fs[-1]))
    x, fs, _ = O.minimize(O.GD(0.1), f, g, [1.0, 1e-6], 500); r["GD_perturbed_start(1,1e-6)"] = (x.tolist(), float(fs[-1]))
    x, _ = O.newton(f, g, h, [1.0, 0.5], 30); r["Newton_from_(1,0.5)"] = (x.tolist(), float(f(x)))
    x, _ = O.newton(f, g, h, [0.5, 0.1], 30); r["Newton_from_(0.5,0.1)_(near saddle)"] = (x.tolist(), float(f(x)))
    x, fs, _ = O.minimize(O.Adam(0.05), f, g, [1.0, 0.0], 800); r["Adam_exact_axis_start"] = (x.tolist(), float(fs[-1]))
    out["saddle"] = r
    json.dump(out, open(RES + "landscapes.json", "w"), indent=1, default=float)
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    xs = np.linspace(-2.2, 2.2, 400); fd = P.double_well()[0]
    ax[0].plot(xs, [fd([v]) for v in xs], "k"); ax[0].set_title("Non-convex 1-D: local vs global min")
    for x0, (xf, ff) in fin.items(): ax[0].plot([x0], [fd([x0])], "o"); ax[0].annotate(f"→{xf:.2f}", (x0, fd([x0])))
    G = np.linspace(-2, 2, 200); X, Y = np.meshgrid(G, G); Z = X**2 - Y**2 + .25 * Y**4
    ax[1].contourf(X, Y, Z, 30, cmap="viridis"); ax[1].plot(0, 0, "r*", ms=15, label="saddle")
    for st, c in [((1, 0), "w"), ((1, 1e-6), "orange")]:
        p = np.array(st, float); tr = [p.copy()]; gd = O.GD(0.1)
        for _ in range(250): p = gd.step(p, P.saddle()[1]); tr.append(p.copy())
        tr = np.array(tr); ax[1].plot(tr[:, 0], tr[:, 1], "-", c=c, label=f"GD from {st}")
    ax[1].legend(fontsize=7); ax[1].set_title("Saddle: GD gets stuck on the axis")
    fc = P.convex_1d()[0]; xs = np.linspace(-1, 1, 200)
    ax[2].plot(xs, [fc([v]) for v in xs], label="x^4 convex, not strongly"); ax[2].plot(xs, xs**2, label="x^2 strongly convex"); ax[2].legend(); ax[2].set_title("Convex flavours")
    savefig("landscapes.png"); return out

# ================================================ 2. convergence theory: strongly convex / convex ====
def convergence_rates():
    rows = []
    for kappa in [2, 10, 100]:
        f, g, _, A = P.quadratic(kappa, d=10); lr = 2 / (1 + kappa); x0 = np.ones(10)
        _, fs, _ = O.minimize(O.GD(lr), f, g, x0, 400)
        emp = (fs[200] / fs[100]) ** (1 / 100); theo = ((kappa - 1) / (kappa + 1)) ** 2
        rows.append(dict(kappa=kappa, empirical_rate_per_iter=emp, theoretical=theo))
    # convex non-strongly: f=x^4 with GD -> sublinear. f_k * k^(1/2)  ~ const  (f_k ~ 1/k^(1/2)... check slope)
    f, g, _ = P.convex_1d(); _, fs, _ = O.minimize(O.GD(0.05), f, g, [1.0], 20000)
    k = np.array([100, 1000, 10000]); slope = np.polyfit(np.log(k), np.log(fs[k]), 1)[0]
    rows.append(dict(kappa="convex x^4 (not strongly)", empirical_rate_per_iter=f"f_k ~ k^{slope:.2f} (sublinear)", theoretical="upper bound O(1/k); x^4 is flatter than worst case so observed decay is faster"))
    # non-convex: only stationarity guaranteed: min_k ||grad|| shrinks like 1/sqrt(k)
    f, g, h = P.rosenbrock(); _, fs, gn = O.minimize(O.GD(2e-3), f, g, [-1.2, 1.0], 20000)
    rows.append(dict(kappa="nonconvex Rosenbrock", empirical_rate_per_iter=f"min||grad|| at k=100: {gn[:101].min():.3g}, k=20000: {gn.min():.3g}", theoretical="only min_k||g||=O(1/sqrt k); no global guarantee"))
    with open(RES + "convergence_rates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
    return rows

# ================================================================= 3. benchmark on 3 objectives ====
def benchmark():
    rows, curves = [], {}
    problems = {}
    f, g, h, _ = P.quadratic(100, d=10); problems["quadratic_kappa100"] = (f, g, h, np.ones(10), 300, 0.0)
    f, g, h = P.rosenbrock(); problems["rosenbrock"] = (f, g, h, np.array([-1.2, 1.0]), 5000, 0.0)
    d = load_breast_cancer(); Xs = StandardScaler().fit_transform(d.data); lg = P.Logistic(Xs, d.target, 1e-3)
    th0 = np.zeros(Xs.shape[1] + 1)
    ref = O.newton(lg.loss, lg.grad, lg.hess, th0, 30, line_search=True)[1][-1]       # reference optimum
    problems["logreg_breast_cancer"] = (lg.loss, lg.grad, lg.hess, th0, 300, ref)
    for pname, (f, g, h, x0, steps, fstar) in problems.items():
        curves[pname] = {}
        for name in OPTS:
            lr, fs, curve = tune(name, f, g, x0, steps)
            # re-time the best config alone (tuning excluded)
            rt = median_time(lambda: O.minimize(O.FIRST_ORDER[name](lr), f, g, x0, steps))
            good = sum(1 for v in curve.values() if np.isfinite(v) and v - fstar < 0.1 * (fs[0] - fstar) + 1e-9)
            edge = lr in (LRS[0], LRS[-1])
            rows.append(dict(problem=pname, method=name, best_lr=lr, best_lr_at_grid_edge=edge, final_gap=fs[-1] - fstar, runtime_s=rt,
                             lr_values_reaching_90pct_gap_reduction=good, n_lr=len(LRS), evals=steps))
            curves[pname][name] = (fs - fstar, curve)
        xN, fsN = O.newton(f, g, h, x0, 100, line_search=True); rt = median_time(lambda: O.newton(f, g, h, x0, 100, line_search=True))
        rows.append(dict(problem=pname, method="Newton(+linesearch)", best_lr="-", final_gap=fsN[-1] - fstar, runtime_s=rt,
                         best_lr_at_grid_edge="-", lr_values_reaching_90pct_gap_reduction="-", n_lr="-", evals=len(fsN) - 1))
        curves[pname]["Newton"] = (fsN - fstar, None)
        xC, fsC = O.newton_cg(f, g, x0, 100); rt = median_time(lambda: O.newton_cg(f, g, x0, 100))
        rows.append(dict(problem=pname, method="Newton-CG (HVP only)", best_lr="-", final_gap=fsC[-1] - fstar, runtime_s=rt,
                         best_lr_at_grid_edge="-", lr_values_reaching_90pct_gap_reduction="-", n_lr="-", evals=len(fsC) - 1))
        curves[pname]["Newton-CG"] = (fsC - fstar, None)
    with open(RES + "benchmark.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for j, (pname, cs) in enumerate(curves.items()):
        for name, (gap, curve) in cs.items():
            ax[0, j].semilogy(np.maximum(gap, 1e-16), label=name)
            if curve: ax[1, j].loglog(list(curve), [max(v - 0, 1e-16) if np.isfinite(v) else np.nan for v in curve.values()], "o-", ms=3, label=name)
        ax[0, j].set_title(f"{pname}: gap to optimum"); ax[0, j].set_xlabel("iteration"); ax[1, j].set_title("LR sensitivity (final loss vs lr)"); ax[1, j].set_xlabel("learning rate")
    ax[0, 0].legend(fontsize=7); savefig("benchmark.png"); return rows

# ========================================================= 4. SGD / mini-batch / noisy objective ====
def sgd_study(seeds=5):
    d = load_breast_cancer(); Xs = StandardScaler().fit_transform(d.data); lg = P.Logistic(Xs, d.target, 1e-3)
    n = len(d.target); th0 = np.zeros(Xs.shape[1] + 1)
    ref = O.newton(lg.loss, lg.grad, lg.hess, th0, 30, line_search=True)[1][-1]
    budget = 20 * n                                          # equal budget: 20 epochs of per-sample gradient work
    rows = []
    for bs in [1, 8, 32, 128, n]:
        for lr in [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
            gaps = []
            for s in range(seeds):
                rng = np.random.default_rng(s); th = th0.copy(); done = 0
                while done < budget:
                    idx = rng.choice(n, size=bs, replace=False) if bs < n else None
                    th = th - lr * lg.grad(th, idx); done += bs
                gaps.append(lg.loss(th) - ref if np.isfinite(lg.loss(th)) else np.inf)
            rows.append(dict(batch=bs, lr=lr, gap_mean=np.mean(gaps), gap_std=np.std(gaps), updates=budget // bs, seeds=seeds))
    with open(RES + "sgd_minibatch.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
    # noise-floor: constant vs decaying step on a noisy quadratic
    f, g, _, A = P.quadratic(10, d=10); fl = {}
    for label, sched in [("const lr=0.05", lambda t: 0.05), ("decay lr=0.5/(1+0.05t)", lambda t: 0.5 / (1 + 0.05 * t))]:
        runs = []
        for s in range(seeds):
            rng = np.random.default_rng(100 + s); x = np.ones(10); h = []
            for t in range(3000): x = x - sched(t) * (A @ x + rng.normal(0, 1.0, 10)); h.append(f(x))
            runs.append(h)
        fl[label] = np.mean(runs, 0)
    plt.figure(figsize=(6, 4))
    for k, v in fl.items(): plt.semilogy(v, label=k)
    plt.title("Noisy gradients: constant step stalls at a noise floor"); plt.xlabel("iteration"); plt.ylabel("f(x)"); plt.legend(); savefig("noisy_sgd.png")
    json.dump({k: float(np.mean(v[-500:])) for k, v in fl.items()}, open(RES + "noise_floor.json", "w"), indent=1)
    return rows, {k: float(np.mean(v[-500:])) for k, v in fl.items()}

# ============================================= 5. backprop MLP + regularisation + Gauss-Newton ====
def mlp_and_gn():
    X, y = make_moons(300, noise=0.2, random_state=0); out = {}
    Xte, yte = make_moons(1000, noise=0.2, random_state=1)
    rows = []
    for l2 in [0.0, 1e-3, 1e-2, 1e-1]:
        tr, te, wn, tl = [], [], [], []
        for seed in range(5):                                  # 5 random initialisations
            m = P.MLP(X, y, hidden=16, l2=l2, seed=seed); mt = P.MLP(Xte, yte, hidden=16, l2=0.0, seed=seed)
            th, fs, _ = O.minimize(O.Adam(0.02), m.loss, m.grad, m.theta0, 3000)
            tr.append(m.accuracy(th)); te.append(mt.accuracy(th)); wn.append(np.linalg.norm(th)); tl.append(fs[-1])
        rows.append(dict(l2=l2, train_loss_incl_reg=np.mean(tl), train_acc=np.mean(tr), test_acc_mean=np.mean(te), test_acc_std=np.std(te), weight_norm=np.mean(wn), seeds=5))
    with open(RES + "mlp_regularisation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
    m = P.MLP(X, y, hidden=8, l2=1e-3, seed=0); curves = {}
    for name, lr in [("GD", 0.5), ("Momentum", 0.1), ("Adam", 0.05), ("RMSProp", 0.01)]:
        _, fs, _ = O.minimize(O.FIRST_ORDER[name](lr), m.loss, m.grad, m.theta0, 1500); curves[name] = fs
    plt.figure(figsize=(6, 4))
    for k, v in curves.items(): plt.semilogy(v, label=k)
    plt.title("MLP (manual backprop), moons"); plt.xlabel("iteration"); plt.ylabel("loss"); plt.legend(); savefig("mlp_backprop.png")
    # Gauss-Newton vs GD vs Newton on nonlinear least squares  y = a exp(b t) + noise
    rng = np.random.default_rng(0); t = np.linspace(0, 2, 40); true = np.array([2.0, -1.3]); y_ = true[0] * np.exp(true[1] * t) + rng.normal(0, 0.02, 40)
    r = lambda p: p[0] * np.exp(p[1] * t) - y_
    J = lambda p: np.stack([np.exp(p[1] * t), p[0] * t * np.exp(p[1] * t)], 1)
    f = lambda p: 0.5 * r(p) @ r(p); g = lambda p: J(p).T @ r(p)
    def H(p):                                              # exact Hessian = J^T J + sum r_i * Hess(r_i)
        e = np.exp(p[1] * t); S = np.zeros((2, 2)); S[0, 1] = S[1, 0] = (r(p) * t * e).sum(); S[1, 1] = (r(p) * p[0] * t**2 * e).sum()
        return J(p).T @ J(p) + S
    p0 = np.array([1.0, -0.5])
    xg, fg = O.gauss_newton(r, J, p0, 20); xn, fn = O.newton(f, g, H, p0, 20, line_search=True)
    xlm, flm = O.gauss_newton(r, J, p0, 20, lam=1e-3)
    gdres = {}
    for budget in (20, 2000):                                    # GD with its lr TUNED for each iteration budget
        gdres[budget] = min(O.minimize(O.GD(lr), f, g, p0, budget)[1][-1] for lr in LRS if np.isfinite(O.minimize(O.GD(lr), f, g, p0, budget)[1][-1]))
    gn = dict(gauss_newton=dict(final=float(fg[-1]), iters_to_1e6=int(np.argmax(fg < fg[-1] + 1e-6)), params=xg.tolist()),
              newton=dict(final=float(fn[-1]), iters_to_1e6=int(np.argmax(fn < fn[-1] + 1e-6)), params=xn.tolist()),
              gd_tuned_lr=dict(final_loss_after_20_iters=float(gdres[20]), final_loss_after_2000_iters=float(gdres[2000])), LM_lam1e3_final=float(flm[-1]), noise_sd=0.02, note='small-residual problem: GN ~ Newton here; GN can fail when residuals are large', true_params=true.tolist())
    json.dump(gn, open(RES + "gauss_newton.json", "w"), indent=1)
    return rows, gn

# =================================================================== 6. curvature approximations ====
def curvature():
    d = load_breast_cancer(); Xs = StandardScaler().fit_transform(d.data); lg = P.Logistic(Xs, d.target, 1e-3)
    th0 = np.zeros(Xs.shape[1] + 1); out = {}
    th = O.minimize(O.GD(0.1), lg.loss, lg.grad, th0, 50)[0]                              # a point part-way down
    v = np.random.default_rng(0).normal(size=len(th)); Hexact = lg.hess(th)
    out["hvp_error_fd_vs_exact"] = float(np.linalg.norm(O.hvp_fd(lg.grad, th, v) - Hexact @ v) / np.linalg.norm(Hexact @ v))
    # Fisher: for logistic regression with the model's own predictive distribution, Fisher == Gauss-Newton Hessian
    ps = P.sigmoid(Xs @ th[:-1] + th[-1]); Xa = np.hstack([Xs, np.ones((len(Xs), 1))])
    F = (Xa * (ps * (1 - ps))[:, None]).T @ Xa / len(Xs)
    Gs = lg.per_sample_grads(th); Femp = Gs.T @ Gs / len(Xs)                                # EMPIRICAL Fisher (uses true labels)
    out["true_Fisher_vs_Hessian_(l2 removed)_rel_err"] = float(np.linalg.norm(F - (Hexact - np.diag(np.r_[1e-3 * np.ones(len(th) - 1), 0]))) / np.linalg.norm(F))
    out["empirical_Fisher_vs_true_Fisher_rel_err"] = float(np.linalg.norm(Femp - F) / np.linalg.norm(F))
    # diagonal / low-rank approximations of the Hessian: how much of it do they capture?
    out["diag_approx_rel_err"] = float(np.linalg.norm(Hexact - np.diag(np.diag(Hexact))) / np.linalg.norm(Hexact))
    w, V = np.linalg.eigh(Hexact); out["condition_number_H"] = float(w[-1] / w[0])
    for r in [1, 3, 5, 10]:
        Hr = (V[:, -r:] * w[-r:]) @ V[:, -r:].T; out[f"lowrank_{r}_rel_err"] = float(np.linalg.norm(Hexact - Hr) / np.linalg.norm(Hexact))
    # does using it help? precondition GD by diagonal Hessian / diag empirical-Fisher (+ damping) vs plain GD vs Newton
    def run_precond(Pinv, lr, steps=60):
        x = th0.copy(); h = [lg.loss(x)]
        for _ in range(steps): x = x - lr * Pinv(x) * lg.grad(x); h.append(lg.loss(x))
        return np.array(h)
    ref = O.newton(lg.loss, lg.grad, lg.hess, th0, 30, line_search=True)[1][-1]
    best = lambda fn: min(((run_precond(*fn(lr)) , lr) for lr in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]), key=lambda a: a[0][-1] if np.isfinite(a[0][-1]) else 9e9)
    plain = best(lambda lr: (lambda x: np.ones(len(x)), lr))
    diagH = best(lambda lr: (lambda x: 1 / (np.diag(lg.hess(x)) + 1e-3), lr))
    diagF = best(lambda lr: (lambda x: 1 / (np.mean(lg.per_sample_grads(x)**2, 0) + 1e-3), lr))
    _, fsn = O.newton(lg.loss, lg.grad, lg.hess, th0, 60, line_search=True)
    out["gap_after_60_iters"] = dict(plain_GD=float(plain[0][-1] - ref), diagonal_Hessian_precond=float(diagH[0][-1] - ref),
                                     diagonal_empirical_Fisher_precond=float(diagF[0][-1] - ref), Newton=float(max(fsn[-1] - ref, 0)))
    json.dump(out, open(RES + "curvature.json", "w"), indent=1); return out

if __name__ == "__main__":
    t0 = time.time(); print("landscapes", json.dumps(landscapes(), indent=1, default=float)[:1500])
    print("rates", *convergence_rates(), sep="\n"); b = benchmark()
    print("benchmark"); [print({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in r.items()}) for r in b]
    r, nf = sgd_study(); print("noise floor", nf); r, gn = mlp_and_gn(); print(*r, sep="\n"); print(gn)
    print("curvature", json.dumps(curvature(), indent=1)); print(f"total {time.time()-t0:.0f}s")
