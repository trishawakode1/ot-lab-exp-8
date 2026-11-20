"""Track B runner -> results/*.csv|json, figures/*.png. Deterministic seeds."""
import time, csv, json, itertools, numpy as np, networkx as nx, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import knapsack as K, scheduling as S, qubo as Q

RES, FIG = "results/", "figures/"
def tm(fn, reps=3):
    ts = []
    for _ in range(reps): t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))
def wcsv(name, rows):
    with open(RES + name, "w", newline="") as fh: w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------------------ 1. knapsack ----
def knapsack_study(seeds=5):
    raw, rows = [], []
    for kind in ["uncorrelated", "strongly_correlated"]:
        for n in [20, 60, 200, 1000]:
            for sd in range(seeds):
                v, w, cap = K.make_instance(n, kind, seed=1000 * n + sd)
                (opt, _), t_dp = tm(lambda: K.dp(v, w, cap))                      # DP = exact reference for every instance
                def add(method, val, t, nodes=None, proven=False): raw.append(dict(kind=kind, n=n, seed=sd, method=method, gap=100 * (opt - val) / opt, t=t, nodes=nodes, proven=proven))
                add("DP (exact)", opt, t_dp, None, True)
                (g, _), t = tm(lambda: K.greedy(v, w, cap)); add("greedy ratio", g, t)
                (g2, _), t = tm(lambda: K.greedy(v, w, cap, True)); add("greedy+best single", g2, t)
                (bv, _, nodes, fin), t = tm(lambda: K.branch_and_bound(v, w, cap)); add("Branch&Bound (3M-node cap)", bv, t, nodes, fin)
                (iv, _, ok), t = tm(lambda: K.ilp(v, w, cap, 30)); add("ILP (HiGHS)", iv, t, None, ok)
    for key in dict.fromkeys((r["kind"], r["n"], r["method"]) for r in raw):
        rr = [r for r in raw if (r["kind"], r["n"], r["method"]) == key]; nodes = [r["nodes"] for r in rr if r["nodes"] is not None]
        rows.append(dict(kind=key[0], n=key[1], method=key[2], median_gap_pct=float(np.median([r["gap"] for r in rr])), max_gap_pct=max(r["gap"] for r in rr),
                         median_time_s=float(np.median([r["t"] for r in rr])), max_time_s=max(r["t"] for r in rr), median_nodes=int(np.median(nodes)) if nodes else "",
                         fraction_proven_optimal=float(np.mean([r["proven"] for r in rr])), instances=seeds))
    wcsv("knapsack.csv", rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for kind, a in zip(["uncorrelated", "strongly_correlated"], ax):
        for m in ["DP (exact)", "greedy ratio", "Branch&Bound (3M-node cap)", "ILP (HiGHS)"]:
            rr = [r for r in rows if r["kind"] == kind and r["method"] == m]; a.loglog([r["n"] for r in rr], [r["median_time_s"] for r in rr], "o-", label=m)
        a.set_title(f"Knapsack median runtime, {seeds} instances ({kind})"); a.set_xlabel("n items"); a.set_ylabel("seconds")
    ax[0].legend(); plt.tight_layout(); plt.savefig(FIG + "knapsack_runtime.png", dpi=130); plt.close(); return rows

# ----------------------------------------------------------------------------- 2. scheduling ----
def scheduling_study():
    rng = np.random.default_rng(0); ratios = []; m = 3
    for _ in range(200):
        p = rng.integers(1, 21, 10).astype(float); opt, ok = S.ilp_makespan(p, m); assert ok; ratios.append(S.lpt(p, m)[0] / opt)
    ratios = np.array(ratios); bound = 4 / 3 - 1 / (3 * m); out = dict(
        instances=200, machines=m, jobs=10, lpt_over_opt_mean=float(ratios.mean()), lpt_over_opt_max=float(ratios.max()),
        fraction_lpt_optimal=float((ratios < 1 + 1e-9).mean()), theoretical_bound=bound)
    p = np.array([3, 3, 2, 2, 2.]); out["worst_case_instance_m2"] = dict(lpt=S.lpt(p, 2)[0], opt=S.brute_force(p, 2), ratio=S.lpt(p, 2)[0] / S.brute_force(p, 2), bound=4 / 3 - 1 / 6)
    big = []
    for n, mm in [(20, 4), (40, 5), (80, 8)]:
        p = np.random.default_rng(n).integers(5, 100, n).astype(float); (l, _), tl = tm(lambda: S.lpt(p, mm)); (c, ok), ti = tm(lambda: S.ilp_makespan(p, mm, 10), reps=1)
        big.append(dict(n=n, m=mm, lpt=l, ilp=c, ilp_proven_optimal=ok, lower_bound=max(p.sum() / mm, p.max()), t_lpt=tl, t_ilp=ti))
    out["scaling"] = big; json.dump(out, open(RES + "scheduling.json", "w"), indent=1); return out

# ----------------------------------------------------------------------------------- 3. Max-Cut ----
def maxcut_study():
    out = {}
    for n in [10, 20]:
        G = nx.random_regular_graph(3, n, seed=7); edges = [(i, j, 1.0) for i, j in G.edges()]; qb = Q.maxcut_qubo(n, edges)
        E = qb.energies_all(); opt = -E.min(); optima = [int(k) for k in np.flatnonzero(E <= E.min() + 1e-9)]
        J, h, c = qb.to_ising(); k = optima[0]; s = 1 - 2 * ((k >> np.arange(n)) & 1)
        assert abs(Q.QUBO.ising_energy(J, h, c, s) - E.min()) < 1e-9                      # Ising form agrees at the optimum
        ls, sa = [], []; ev_ls = ev_sa = None
        for seed in range(30):
            rng = np.random.default_rng(seed); ls.append(-qb.local_search(rng, restarts=20)[0]); sa.append(-qb.simulated_annealing(rng, sweeps=200)[0])
        out[f"n{n}"] = dict(edges=len(edges), optimal_cut=float(opt), n_optimal_bitstrings=len(optima), ising_const=float(c),
            local_search=dict(mean_ratio=float(np.mean(ls) / opt), success_rate=float(np.mean(np.array(ls) >= opt - 1e-9)), evals_per_run="20 restarts x steps (~20-100 energy evals each)"),
            simulated_annealing=dict(mean_ratio=float(np.mean(sa) / opt), success_rate=float(np.mean(np.array(sa) >= opt - 1e-9)), flips_proposed_per_run=200 * n),
            brute_force_evals=2**n, seeds=30)
        if n == 10:                                                                            # instance reused by Track C (QAOA)
            json.dump(dict(n=n, edges=[[i, j] for i, j, _ in edges], optimal_cut=float(opt), optimal_bitstrings=optima,
                           ising_J=J.tolist(), ising_h=h.tolist(), ising_const=float(c), bit_order="bit i of integer = x_i"), open(RES + "maxcut_n10_for_trackC.json", "w"))
    # larger graph: no brute force -> "best known" = best over all runs, labelled as such
    n = 60; G = nx.random_regular_graph(3, n, seed=11); qb = Q.maxcut_qubo(n, [(i, j, 1.0) for i, j in G.edges()])
    ls = [-qb.local_search(np.random.default_rng(s), 20)[0] for s in range(30)]; sa = [-qb.simulated_annealing(np.random.default_rng(s), 300)[0] for s in range(30)]
    out["n60_no_exact_reference"] = dict(best_known_from_runs=float(max(ls + sa)), ls_mean=float(np.mean(ls)), ls_std=float(np.std(ls)), sa_mean=float(np.mean(sa)), sa_std=float(np.std(sa)),
                                         note="optimum unknown; 'best known' is only the best value any run found")
    json.dump(out, open(RES + "maxcut.json", "w"), indent=1); return out

# ------------------------------------------------------------------ 4. feature selection as QUBO ----
def feature_selection_study():
    d = load_breast_cancer(); X = d.data[:, :20]; y = d.target; n = 20; lam = 0.2
    rel = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(n)]); red = np.abs(np.corrcoef(X.T)); np.fill_diagonal(red, 0)
    base_q = Q.QUBO(np.diag(-rel) + lam * red / 2)                                 # E = -sum rel_i x_i + lam sum_{i<j} red_ij x_i x_j
    Eb = base_q.energies_all(); idx = np.arange(1 << n); pop = sum(((idx >> i) & 1) for i in range(n))
    m_star = int(pop[int(np.argmin(Eb))]); obj = lambda sel: base_q.energy(np.isin(np.arange(n), sel).astype(int))
    delta = float(max(abs(base_q.M[i, i]) + 2 * np.abs(np.delete(base_q.M[i], i)).sum() for i in range(n)))   # max |energy change| of any single flip
    rows = []; info = dict(n=n, lam=lam, unconstrained_optimum_size=m_star, sufficient_penalty_bound=delta)
    for k in [4, 6, 10]:
        ref = float(Eb[pop == k].min())                                            # exact constrained optimum by full enumeration
        for P in [0.0, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
            qb = Q.add(base_q, Q.cardinality_penalty(n, k, P)); E = qb.energies_all(); kb = int(np.argmin(E)); x = (kb >> np.arange(n)) & 1
            Dp = float(max(abs(qb.M[i, i]) + 2 * np.abs(np.delete(qb.M[i], i)).sum() for i in range(n)))
            res = {}
            for label, (T0, T1) in {"fixedT": (2.0, 0.02), "scaledT": (0.3 * Dp, 0.3 * Dp / 200)}.items():
                sa = [qb.simulated_annealing(np.random.default_rng(s), sweeps=100, T0=T0, T1=T1) for s in range(30)]
                res[label + "_hits_global_min"] = float(np.mean([abs(e - E.min()) < 1e-9 for e, _ in sa])); res[label + "_feasible"] = float(np.mean([xx.sum() == k for _, xx in sa]))
            rows.append(dict(k=k, P=P, P_exceeds_sufficient_bound=P > delta, exact_n_selected=int(x.sum()), exact_feasible=bool(x.sum() == k),
                             exact_unpenalised_obj=float(base_q.energy(x)), constrained_optimum=ref, **res))
    wcsv("feature_qubo_penalty.csv", rows)
    # downstream usefulness at k=6 (5-fold CV logistic regression), with spread
    k = 6; ref6 = float(Eb[pop == k].min()); exact_sel = [i for i in range(n) if (int(np.argmin(np.where(pop == k, Eb, np.inf))) >> i) & 1]
    sel_g = []
    for _ in range(k): sel_g.append(min((i for i in range(n) if i not in sel_g), key=lambda i: obj(sel_g + [i])))
    top_rel = list(np.argsort(-rel)[:k]); rng0 = np.random.default_rng(0)
    cvs = lambda cols: cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X[:, list(cols)], y, cv=5)
    rnd = [cvs(rng0.choice(n, k, replace=False)).mean() for _ in range(20)]
    info["k6"] = dict(exact_selection=exact_sel, exact_value=ref6, greedy_selection=sorted(sel_g), greedy_value=obj(sel_g), greedy_equals_exact=sorted(sel_g) == sorted(exact_sel),
                      cv_acc={"exact QUBO optimum": [float(cvs(exact_sel).mean()), float(cvs(exact_sel).std())], "top-6 relevance only": [float(cvs(top_rel).mean()), float(cvs(top_rel).std())],
                              "random 6 (20 draws, mean/std over draws)": [float(np.mean(rnd)), float(np.std(rnd))], "all 20": [float(cvs(range(n)).mean()), float(cvs(range(n)).std())]})
    kg = {}
    for kk in [4, 6, 10]:
        sg = []
        for _ in range(kk): sg.append(min((i for i in range(n) if i not in sg), key=lambda i: obj(sg + [i])))
        kg[kk] = dict(greedy=obj(sg), exact=float(Eb[pop == kk].min()))
    info["greedy_vs_exact_objective_by_k"] = kg
    sw = {}
    for kk in [4, 6, 10]:                                                          # penalty-free, constraint-preserving SA on the ORIGINAL objective
        ref = float(Eb[pop == kk].min()); runs = [Q.swap_annealing(base_q, np.random.default_rng(s), kk, sweeps=100) for s in range(30)]
        sw[kk] = dict(hits_constrained_optimum=float(np.mean([abs(e - ref) < 1e-9 for e, _ in runs])), all_feasible=bool(all(x.sum() == kk for _, x in runs)), energy_evals=100 * n)
    info["swap_move_SA_no_penalty"] = sw; json.dump(info, open(RES + "feature_selection.json", "w"), indent=1)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for j, k in enumerate([4, 6, 10]):
        rr = [r for r in rows if r["k"] == k]; xs = range(len(rr)); lab = [r["P"] for r in rr]
        ax[0, j].plot(xs, [r["exact_n_selected"] for r in rr], "o-"); ax[0, j].axhline(k, ls="--", c="r", label=f"required k={k}"); ax[0, j].axhline(m_star, ls=":", c="gray", label="unconstrained optimum size")
        ax[0, j].set_title(f"k={k}: #features in exact QUBO optimum"); ax[0, j].set_xticks(list(xs)); ax[0, j].set_xticklabels(lab); ax[0, j].legend(fontsize=7)
        ax[1, j].plot(xs, [r["fixedT_hits_global_min"] for r in rr], "o-", label="SA fixed T: hits optimum"); ax[1, j].plot(xs, [r["scaledT_hits_global_min"] for r in rr], "s-", label="SA energy-scaled T: hits optimum")
        ax[1, j].plot(xs, [r["scaledT_feasible"] for r in rr], "^--", label="scaled T: feasible"); ax[1, j].set_xticks(list(xs)); ax[1, j].set_xticklabels(lab); ax[1, j].set_xlabel("penalty P"); ax[1, j].set_ylim(-.05, 1.05)
    ax[1, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig(FIG + "qubo_penalty.png", dpi=130); plt.close(); return rows, info

if __name__ == "__main__":
    t = time.time(); r = knapsack_study()
    for x in r: print(f"{x['kind'][:6]} n={x['n']:4d} {x['method']:23s} gap med/max={x['median_gap_pct']:.3f}/{x['max_gap_pct']:.3f}% t med/max={x['median_time_s']*1e3:8.2f}/{x['max_time_s']*1e3:8.2f}ms nodes={x['median_nodes']} proven={x['fraction_proven_optimal']}")
    print(json.dumps(scheduling_study(), indent=1)); print(json.dumps(maxcut_study(), indent=1))
    rows, fs = feature_selection_study(); [print(x) for x in rows]; print(json.dumps(fs, indent=1)); print(f"total {time.time()-t:.0f}s")
