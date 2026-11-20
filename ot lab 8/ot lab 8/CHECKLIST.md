# Chapter 6 coverage checklist

> **TODO for you:** the assignment says this list must match the *exact* Chapter 6 syllabus taught in class.
> Compare it with your lecture slides and add/remove rows. Status: ✅ done & verified · 🔶 partial · ⬜ not started.

| # | Chapter 6 item | Track | Experiment / code | Result location | Status |
|---|---|---|---|---|---|
| 1 | ML as optimization (params, loss, regularization) | A | `problems.py` Logistic/MLP, `mlp_and_gn()` | `results/mlp_regularisation.csv` | ✅ |
| 2 | Continuous vs discrete optimization | A, B | A: continuous; B: discrete | track READMEs | ✅ |
| 3 | Convex vs non-convex; local/global minima; saddle | A | `landscapes()` | `results/landscapes.json`, `figures/landscapes.png` | ✅ |
| 4 | Basic quantum concepts, role of optimization in QML | C, I | — | — | ⬜ |
| 5 | GD, SGD, mini-batch SGD | A | `optimizers.GD`, `sgd_study()` | `results/benchmark.csv`, `sgd_minibatch.csv` | ✅ |
| 6 | Momentum, NAG | A | `optimizers.Momentum/NAG` | `results/benchmark.csv` | ✅ |
| 7 | AdaGrad, RMSProp, Adam | A | `optimizers.py` | `results/benchmark.csv` | ✅ |
| 8 | Newton, Gauss–Newton | A | `optimizers.newton/gauss_newton` | `benchmark.csv`, `gauss_newton.json` | ✅ |
| 9 | Backpropagation (distinct from optimizer) | A | `problems.MLP.grad` + finite-difference test | `test_track_a.py` | ✅ |
| 10 | Curvature approx: Fisher, diagonal, low-rank, HVP | A | `curvature()`, `optimizers.hvp_fd/newton_cg` | `results/curvature.json` | ✅ |
| 11 | Convergence: convex / strongly convex / non-convex | A | `convergence_rates()` | `results/convergence_rates.csv` | ✅ |
| 12 | Step-size & optimizer tuning; noisy objectives | A | `tune()`, `sgd_study()` | `benchmark.csv`, `noise_floor.json` | ✅ |
| 13 | Greedy, ILP, Branch-and-Bound, DP | B | `knapsack.py`, `scheduling.py` | `results/knapsack.csv`, `scheduling.json` | ✅ |
| 14 | Feature selection / combinatorial apps | B | `run_track_b.py: feature_selection_study()` | `feature_selection.json` | ✅ |
| 15 | QUBO / Ising (binary↔spin, penalties) | B | `qubo.py` | `maxcut.json`, `feature_qubo_penalty.csv` | ✅ (annealer-embedding limits: Track C) |
| 16 | QAOA, VQE, PQC/QNN | C | — | — | ⬜ |
| 17 | Parameter-shift, SPSA, quantum natural gradient | C | — | — | ⬜ |
| 18 | Deutsch–Jozsa, Bernstein–Vazirani, Grover, QFT, QPE, Shor, annealing | I | — | — | ⬜ |
| 19 | Variational classifier, quantum kernel | I | — | — | ⬜ |
| 20 | Barren plateaus, finite shots, noise, QUBO-on-annealer limits | C, I | — | — | ⬜ |
| 21 | Submodular: diminishing returns, greedy, guarantee conditions | D | — | — | ⬜ |
| 22 | Parallel: load balancing, makespan, work stealing, locality, Amdahl/Gustafson, data/model/pipeline | D | — | — | ⬜ |
| 23 | Strategy selection | E | — | — | ⬜ |
| 24 | FedAvg, MGDA, GradNorm | D | — | — | ⬜ |
