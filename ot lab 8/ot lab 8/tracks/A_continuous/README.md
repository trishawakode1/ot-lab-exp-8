# Track A — Continuous optimization

**Run:** `cd tracks/A_continuous && python -m unittest test_track_a -v && python run_track_a.py && python make_tables.py`
(≈40 s, numpy only, fixed seeds). All numbers below come from `results/` (full tables: `results/tables.md`).
Hardware: sandbox CPU, 1 core. Timings are medians of 7 runs of millisecond-scale numpy code, so they mostly reflect Python overhead.

## Objective, baseline, fairness protocol
Minimise a loss f(θ) over continuous θ. Baseline = plain GD. For every first-order method the learning rate was tuned on the **same 13-point log grid (1e-5…1e2)**
at the **same iteration budget**; the best-lr-at-grid-edge flag is reported (it was `False` for all benchmark rows). Newton / Newton-CG are
reported separately because one iteration costs far more than a gradient step.
Test problems: ill-conditioned quadratic (κ=100, strongly convex), Rosenbrock (non-convex), L2 logistic regression on breast-cancer data (convex ML task).

## What the experiments show (and how sure we are)
| Finding | Evidence | Confidence / caveat |
|---|---|---|
| GD's rate on a quadratic equals ((κ−1)/(κ+1))² | empirical vs theory agree to ~1e-15 for κ=2,10,100 (`convergence_rates.csv`) | exact, deterministic |
| Newton solves a quadratic in 1 step independent of κ | gap 1e-29, 1 iteration | exact |
| Momentum/NAG/Adam beat plain GD at equal iterations on all 3 problems | e.g. κ=100: GD 3e-3, NAG 6e-17, Adam 6e-13 | one problem instance each; tuned lr per method |
| Newton needs ~1–22 iterations vs 300–5000 first-order | `benchmark.csv` | each Newton iteration costs a d×d solve; fine for d≤31 here, not for large models |
| Newton-CG (Hessian-vector products only) matches Newton but was **slower** in wall-clock on logistic regression (157 ms vs 3.9 ms) | `benchmark.csv` | finite-difference HVP in Python; the advantage is memory, not speed, at d=31 |
| Newton converges to the **saddle** of x²−y²+¼y⁴ | `landscapes.json` | demonstrates attraction to any stationary point |
| GD started exactly on the saddle's stable axis never escapes; a 1e-6 perturbation escapes to f=−1 | `landscapes.json`, figure | deterministic |
| Different starts reach different minima of a double well (f=−3.51 global vs −1.07 local) | `landscapes.json` | |
| At equal sample budget SGD with batch 1–128 reach similar gaps (≈2–3e-3) once lr is tuned; full batch is worse (9.7e-3) | `tables.md` | **single dataset, 5 seeds**; the batch-size ranking is NOT significant among 1–128 |
| Constant step on noisy gradients stalls at a noise floor (f≈0.14) vs decaying step (f≈0.009) | `noise_floor.json` | synthetic noise, 5 seeds |
| L2 = 1e-3 improves MLP test accuracy (0.964±0.001 vs 0.950±0.005 unregularised); L2 ≥ 1e-2 underfits | `mlp_regularisation.csv` | 5 inits, 1 dataset; L2 also penalises biases |
| Backprop gradient matches finite differences (rel. err < 1e-6) | `test_track_a.py` | |
| For logistic regression the true Fisher equals the (data-part of the) Hessian (err 1.6e-17); the **empirical** Fisher is 64% off | `curvature.json` | holds for canonical-link GLMs only |
| Diagonal Hessian is a poor Frobenius approximation (87% error) yet a diagonal preconditioner still helps (gap 1.1e-3 vs 3.0e-3 plain GD after 60 iters; diagonal empirical Fisher 3.4e-4) | `curvature.json` | single run; lr tuned on 8-point grid; Frobenius error ≠ preconditioner quality |
| Gauss–Newton ≈ Newton ≈ **tuned** GD (20 iters) on the exponential fit | `gauss_newton.json` | small-residual, 2-parameter problem — this is NOT evidence GN is useless in general |

## Limitations (read before claiming anything)
* Optimizer ranking is problem-specific; three problems and tuned-on-the-same-objective lr give an optimistic view of every method.
* Adam's lr-sensitivity on Rosenbrock is erratic (also "converges" at lr=10); treat it as a lucky setting, not robustness.
* Runtime differences between first-order methods (2–4 ms) are within Python overhead, not algorithmic evidence.
* Convergence theory is verified only for GD on quadratics; Momentum/NAG rates are measured, not derived here.
* Not covered here: GradNorm/MGDA/FedAvg (Track D), quantum optimizers (Track C).

## Viva notes (be able to explain, without AI)
1. Why does GD's rate depend on κ, and why does Newton not? 2. Why does the exact-axis start stay on the saddle? 3. Backprop vs optimizer: what does `MLP.grad` do layer by layer?
4. Why is Gauss–Newton `JᵀJ` and when does dropping the second-derivative term hurt? 5. Why is the empirical Fisher not the Hessian? 6. Why does constant-step SGD plateau?
