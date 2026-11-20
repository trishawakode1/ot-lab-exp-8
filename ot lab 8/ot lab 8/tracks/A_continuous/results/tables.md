
**quadratic_kappa100**

| method | best_lr | final_gap | runtime_s | best_lr_at_grid_edge | lr_values_reaching_90pct_gap_reduction |
|---|---|---|---|---|---|
| GD | 0.01 | 3.17e-03 | 2.47 ms | False | 5 |
| Momentum | 0.01 | 3.76e-12 | 2.79 ms | False | 8 |
| NAG | 0.00316 | 5.61e-17 | 3.14 ms | False | 7 |
| AdaGrad | 1 | 3.14e-06 | 3.19 ms | False | 8 |
| RMSProp | 0.01 | 1.85e-02 | 3.57 ms | False | 5 |
| Adam | 0.316 | 6.08e-13 | 4.54 ms | False | 10 |
| Newton(+linesearch) | - | 1.26e-29 | 0.03 ms | - | - |
| Newton-CG (HVP only) | - | 1.32e-24 | 0.24 ms | - | - |

**rosenbrock**

| method | best_lr | final_gap | runtime_s | best_lr_at_grid_edge | lr_values_reaching_90pct_gap_reduction |
|---|---|---|---|---|---|
| GD | 0.001 | 3.76e-03 | 32.46 ms | False | 4 |
| Momentum | 0.00316 | 1.24e-30 | 43.84 ms | False | 6 |
| NAG | 0.001 | 1.38e-19 | 43.49 ms | False | 5 |
| AdaGrad | 1 | 6.88e-05 | 46.13 ms | False | 5 |
| RMSProp | 0.001 | 3.10e-04 | 46.21 ms | False | 7 |
| Adam | 0.0316 | 1.69e-27 | 56.88 ms | False | 9 |
| Newton(+linesearch) | - | 0.00e+00 | 0.37 ms | - | - |
| Newton-CG (HVP only) | - | 3.29e-21 | 1.05 ms | - | - |

**logreg_breast_cancer**

| method | best_lr | final_gap | runtime_s | best_lr_at_grid_edge | lr_values_reaching_90pct_gap_reduction |
|---|---|---|---|---|---|
| GD | 10 | 5.88e-07 | 24.83 ms | False | 6 |
| Momentum | 3.16 | 1.80e-13 | 25.82 ms | False | 8 |
| NAG | 3.16 | 3.51e-15 | 24.31 ms | False | 8 |
| AdaGrad | 1 | 1.42e-04 | 24.55 ms | False | 6 |
| RMSProp | 0.01 | 3.62e-05 | 25.21 ms | False | 6 |
| Adam | 1 | 2.69e-13 | 26.02 ms | False | 7 |
| Newton(+linesearch) | - | 0.00e+00 | 2.66 ms | - | - |
| Newton-CG (HVP only) | - | -2.78e-17 | 162.14 ms | - | - |

**Convergence rates**

| kappa | empirical_rate_per_iter | theoretical |
|---|---|---|
| 2 | 0.11111111111111127 | 0.1111111111111111 |
| 10 | 0.6694214876033057 | 0.6694214876033059 |
| 100 | 0.960788158023723 | 0.9607881580237231 |
| convex x^4 (not strongly) | f_k ~ k^-1.98 (sublinear) | upper bound O(1/k); x^4 is flatter than worst case so observed decay is faster |
| nonconvex Rosenbrock | min||grad|| at k=100: 1.84, k=20000: 8.81e-06 | only min_k||g||=O(1/sqrt k); no global guarantee |

**SGD / mini-batch (equal budget = 20 epochs of per-sample gradients, best lr per batch size, 5 seeds)**

| batch | lr | gap_mean | gap_std | updates |
|---|---|---|---|---|
| 1 | 0.03 | 2.29e-03 | 5.29e-04 | 11380 |
| 8 | 0.3 | 2.05e-03 | 4.55e-04 | 1422 |
| 32 | 1 | 2.58e-03 | 5.72e-04 | 355 |
| 128 | 3 | 3.03e-03 | 5.58e-04 | 88 |
| 569 | 3 | 9.68e-03 | 0.00e+00 | 20 |

**Noisy objective, final f(x) (mean of last 500 iters, 5 seeds)**
```
{
 "const lr=0.05": 0.1420844671940282,
 "decay lr=0.5/(1+0.05t)": 0.008690377153490513
}```

**MLP with manual backprop: L2 regularisation (5 inits)**

| l2 | train_acc | test_acc_mean | test_acc_std | weight_norm |
|---|---|---|---|---|
| 0.0 | 0.998 | 0.9498 | 0.005154 | 45.32 |
| 0.001 | 0.9833 | 0.9636 | 0.0008 | 10.24 |
| 0.01 | 0.86 | 0.8772 | 0.0204 | 3.188 |
| 0.1 | 0.8267 | 0.8188 | 0.0004 | 1.726 |

**Gauss-Newton / Newton / tuned GD**
```
{
 "gauss_newton": {
  "final": 0.004870010328437049,
  "iters_to_1e6": 3,
  "params": [
   2.001921293924565,
   -1.30659285138197
  ]
 },
 "newton": {
  "final": 0.004870010328437041,
  "iters_to_1e6": 3,
  "params": [
   2.0019212929729258,
   -1.3065928506998148
  ]
 },
 "gd_tuned_lr": {
  "final_loss_after_20_iters": 0.004878613767282806,
  "final_loss_after_2000_iters": 0.004870010328437051
 },
 "LM_lam1e3_final": 0.004870010328437049,
 "noise_sd": 0.02,
 "note": "small-residual problem: GN ~ Newton here; GN can fail when residuals are large",
 "true_params": [
  2.0,
  -1.3
 ]
}
```

**Curvature approximations**
```
{
 "hvp_error_fd_vs_exact": 9.64917908993908e-11,
 "true_Fisher_vs_Hessian_(l2 removed)_rel_err": 1.638838002085513e-17,
 "empirical_Fisher_vs_true_Fisher_rel_err": 0.6426919145915498,
 "diag_approx_rel_err": 0.8691720922085246,
 "condition_number_H": 494.315565928943,
 "lowrank_1_rel_err": 0.5810261552167517,
 "lowrank_3_rel_err": 0.3711309448383642,
 "lowrank_5_rel_err": 0.21040486879499187,
 "lowrank_10_rel_err": 0.0616671719757491,
 "gap_after_60_iters": {
  "plain_GD": 0.0029850778781451565,
  "diagonal_Hessian_precond": 0.0010958127315937038,
  "diagonal_empirical_Fisher_precond": 0.0003429588280353599,
  "Newton": 0.0
 }
}
```