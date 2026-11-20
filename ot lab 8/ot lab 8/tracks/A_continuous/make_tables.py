"""Builds results/tables.md from the CSV/JSON outputs so the report never contains hand-typed numbers."""
import csv, json
R = "results/"; out = []
def md(rows, cols, fmt=None):
    fmt = fmt or {}
    o = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows: o.append("| " + " | ".join((fmt.get(c, lambda v: v)(r[c]) if c in r else "") for c in cols) + " |")
    return "\n".join(o)
g = lambda v: f"{float(v):.2e}" if v not in ("-", "") else v
b = list(csv.DictReader(open(R + "benchmark.csv")))
for p in dict.fromkeys(r["problem"] for r in b):
    out += [f"\n**{p}**\n", md([r for r in b if r["problem"] == p], ["method", "best_lr", "final_gap", "runtime_s", "best_lr_at_grid_edge", "lr_values_reaching_90pct_gap_reduction"],
                              {"final_gap": g, "runtime_s": lambda v: f"{float(v)*1e3:.2f} ms", "best_lr": lambda v: v if v == "-" else f"{float(v):.3g}"})]
out += ["\n**Convergence rates**\n", md(list(csv.DictReader(open(R + "convergence_rates.csv"))), ["kappa", "empirical_rate_per_iter", "theoretical"])]
s = list(csv.DictReader(open(R + "sgd_minibatch.csv"))); best = []
for bs in dict.fromkeys(int(r["batch"]) for r in s):
    r = min([x for x in s if int(x["batch"]) == bs], key=lambda x: float(x["gap_mean"])); best.append(r)
out += ["\n**SGD / mini-batch (equal budget = 20 epochs of per-sample gradients, best lr per batch size, 5 seeds)**\n", md(best, ["batch", "lr", "gap_mean", "gap_std", "updates"], {"gap_mean": g, "gap_std": g})]
out += ["\n**Noisy objective, final f(x) (mean of last 500 iters, 5 seeds)**\n```\n" + open(R + "noise_floor.json").read() + "```"]
out += ["\n**MLP with manual backprop: L2 regularisation (5 inits)**\n", md(list(csv.DictReader(open(R + "mlp_regularisation.csv"))), ["l2", "train_acc", "test_acc_mean", "test_acc_std", "weight_norm"], {k: (lambda v: f"{float(v):.4g}") for k in ["train_acc", "test_acc_mean", "test_acc_std", "weight_norm"]})]
out += ["\n**Gauss-Newton / Newton / tuned GD**\n```\n" + open(R + "gauss_newton.json").read() + "\n```", "\n**Curvature approximations**\n```\n" + open(R + "curvature.json").read() + "\n```"]
open(R + "tables.md", "w").write("\n".join(out)); print("ok")
