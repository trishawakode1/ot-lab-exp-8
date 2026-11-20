# Track B — Discrete optimization

**Run:** `cd tracks/B_discrete && python -m unittest test_track_b -v && python run_track_b.py` (≈2 min, numpy/scipy/networkx/sklearn, fixed seeds).
Full tables: `results/knapsack.csv`, `scheduling.json`, `maxcut.json`, `feature_qubo_penalty.csv`, `feature_selection.json`. Figures: `figures/`.
Hardware: sandbox, 1 CPU core. Timings: median of 3 runs per instance; knapsack statistics over **5 random instances per configuration**.

## Problems and formulations
| Problem | Variables | Objective | Constraint | Methods |
|---|---|---|---|---|
| 0/1 knapsack (budgeted selection) | x∈{0,1}ⁿ | max v·x | w·x ≤ C (C=Σw/2) | greedy (ratio, +best-single), DP, Branch-and-Bound (own code), ILP (HiGHS) |
| Makespan scheduling P‖Cmax | job→machine | min latest machine load | each job on one machine | LPT greedy, ILP, brute force |
| Max-Cut (3-regular graphs, n=10/20/60) | x∈{0,1}ⁿ | max cut = min x᠎ᵀMx | none | QUBO ↔ Ising, brute force, 1-flip local search, simulated annealing |
| Feature selection (20 breast-cancer features) | x∈{0,1}²⁰ | min −Σ relᵢxᵢ + 0.2 Σ redᵢⱼxᵢxⱼ | Σx = k, via penalty P(Σx−k)² | exact enumeration, greedy forward, SA (penalty), SA (swap moves) |

Verified by tests (`test_track_b.py`, 11 tests): all exact knapsack solvers agree with brute force on 40 instances; greedy+best-single is a ½-approximation; the QUBO and Ising energies are
**identical on every bitstring**; Max-Cut energy equals −cut; the cardinality penalty is exactly P(Σx−k)²; the flip-delta formula matches recomputed energies.

## Findings (what the data supports)
| Finding | Evidence | Caveat |
|---|---|---|
| Greedy is near-optimal on easy instances (median gap 0.01–1.0%) but 2.5–2.7% median (max 9.3%) on strongly correlated n=20–60 | `knapsack.csv` | gap shrinks with n here (0.16% at n=1000) — a property of this instance family, not a guarantee |
| Ratio-greedy can get 0.2% of optimum; the "best single item" fix restores the ½ guarantee | unit test (adversarial instance) | the fix did not change results on random instances |
| B&B proves optimality in ≤90 ms (≤1.8k nodes) on uncorrelated instances up to n=1000, but on **strongly correlated** ones proves it for only 60% (n=60), 40% (n=200), 0% (n=1000) of instances within a 3M-node/5 s cap | `knapsack.csv` | even when unproven, its incumbent was within 0.05% median |
| ILP (HiGHS) proved optimality on every instance, ≤ 270 ms; it was fastest of the exact methods on the hard correlated family (7–23 ms median at n=200–1000) | `knapsack.csv` | one solver, default settings; our B&B is a simple Python DFS, not a competitor to HiGHS |
| DP is exact and fast here (≤55 ms) but needs O(n·C) time/memory — pseudo-polynomial | code | capacities were ≤ ~27k; huge C would break it |
| LPT scheduling: mean 1.025× optimum, max 1.143× over 200 random instances (n=10, m=3), optimal on 41%; never exceeded the 4/3−1/(3m)=1.222 bound; worst-case instance reaches the bound exactly (7 vs 6, m=2) | `scheduling.json` | ILP is ~28,000× slower at n=80,m=8 (3.6 s vs 0.13 ms) for a 0.6% better makespan (511 vs 514) |
| QUBO→Ising conversion is exact (const = −7.5 for the saved n=10 graph) | test on all 2⁷ states + optimum check | |
| Max-Cut n=10/20: 1-flip local search and SA both reach the optimum in 30/30 runs. **These instances are too easy to separate the heuristics** | `maxcut.json` | harder instances needed; at n=60 (no exact optimum): local search 80.6±0.7, SA 81.1±0.7, best seen 82 |
| Penalty P too small ⇒ QUBO optimum **violates** the constraint (k=4: 5–6 features at P≤0.1; k=10: 6–9 at P≤0.1); P≥0.3 sufficed empirically | `feature_qubo_penalty.csv`, figure | provable sufficient bound P>2.76 (any flip changes the objective by ≤ Δ=2.76) is ~9× larger than needed |
| Large P makes penalised SA return feasible but **suboptimal** solutions (optimum hit rate ≈0–3% for P≥1) even with an energy-scaled temperature | same | |
| A penalty-free SA with swap moves (constraint built into the move set) hits the constrained optimum 93–100% at the same 2,000-proposal budget | `feature_selection.json` | supports (does not prove) the explanation that large P creates barriers between feasible states |
| Greedy forward selection equals the exact optimum at k=6 and is 0.08% (k=4) / 1.9% (k=10) worse in objective | `feature_selection.json` | the QUBO machinery is not needed to solve this n=20 instance — a negative result for QUBO-as-solver |
| Selected 6 features: CV accuracy 0.931±0.024 vs random-6 0.916 and top-relevance-6 0.912; all 20 features 0.960±0.012 | same | differences ≈ 1 fold-std; weak evidence the proxy objective helps; feature selection **costs** accuracy here |

## Hand-off to Track C
`results/maxcut_n10_for_trackC.json` holds the n=10 graph, optimal cut 13 (2 optimal bitstrings: a complementary pair), the Ising J/h/const and bit-order convention. Classical baselines: brute force (2¹⁰ evals), local search and SA.

## Limitations
* Greedy/B&B/ILP gap and time statistics are specific to the two random instance families used. * Python B&B timing is not comparable to compiled solvers.
* The temperature-scaling rule for SA is a heuristic; the penalty-barrier explanation was tested only through the swap-move comparison.
* Feature-selection objective is a correlation proxy (mRMR-style); one dataset, first 20 columns only.

## Viva notes
1. Why is the fractional relaxation a valid B&B upper bound? 2. Why does DP's complexity not contradict knapsack's NP-hardness? 3. Derive x=(1−s)/2 → J, h, const. 4. Why is P>Δ sufficient for feasibility? 5. Why does large P hurt annealing? 6. Why does correlated knapsack hurt B&B but not DP?
