# Failure log (AI-generated errors, fixes, lessons)

Entries below were observed while building Track A with Claude (claude.ai). **Re-run the code yourself
and confirm each entry; add your own failures from other tools/tracks.** The assignment requires ≥3.

| # | Track | What the AI produced | How it was caught | Fix | Lesson |
|---|---|---|---|---|---|
| 1 | A | Experiment script used `iters_to_1e-6=` as a keyword argument → `SyntaxError` | Ran the script | Renamed to `iters_to_1e6` | Run generated code immediately; never assume it parses. |
| 2 | A | Shell `mkdir -p a/{b,c}` created one folder literally named `{b,c}` (brace expansion not applied) → `FileNotFoundError` when saving results | Runtime error | Removed the folders, created each explicitly | Check the filesystem after setup commands. |
| 3 | A | **Truncated learning-rate grid** (max 0.3) in the SGD study: best lr was at the grid edge for 4 of 5 batch sizes, which made "batch size 1 wins" look true | Reviewing the best-lr column | Widened grid to 30, added `best_lr_at_grid_edge` flag. With a proper grid batch sizes 1–128 are roughly tied (≈2–3e-3 gap) | Always check whether the tuned optimum lies on the edge of the search range. |
| 4 | A | **Unfair baseline**: plain GD with an untuned lr (0.01) was compared with Gauss–Newton, giving GD loss 0.65 vs 0.00487 | Suspicious size of gap | Tuned GD's lr per budget: GD reaches 0.004879 in 20 iterations (≈ GN). | A "second-order is 100× better" claim can be an untuned-baseline artifact. |
| 5 | A | Curvature-preconditioning experiment capped plain GD's lr at 1.0 while the best plain-GD lr elsewhere was 10 | Cross-checking with benchmark results | Extended the same lr grid (up to 30) for all variants | Use one shared, wide tuning protocol for every method. |
| 6 | A | Figure showed a perturbed-start saddle trajectory as "stuck" because only 60 steps were plotted; escape needs ~100 | Compared figure to numeric result (escape to f=-1) | Plot 250 steps | Verify plots against numbers, not the reverse. |
| 7 | A | Result label claimed the x⁴ decay should match "theory −1"; measured slope was −1.98 | Reading the CSV | Reworded: O(1/k) is an *upper bound*; x⁴ is flatter than the worst case | Do not present a bound as the exact rate. |
| 8 | A | Test file contained leftover dead code from drafting | Code review | Removed | Review generated tests, not just generated code. |
| 9 | B | Feature-selection penalty study where the constraint never bound (the unconstrained optimum already had k=6), yet the figure title claimed "Too-small P violates the constraint" | Reading the CSV against the plot title | Redesigned with k=4,6,10 so the constraint binds; added a provable sufficient penalty bound and kept k=6 as a labelled contrast | A title/claim must be checked against the data it sits on. |
| 10 | B | Branch-and-Bound node cap (400k) was just below the ~426k nodes needed to prove optimality, so "not proven" reflected the cap, not the method | Raised the cap in a probe run; measured nodes/µs to rule out a counting bug | Added time + node limit, report fraction proven, rename label to the true cap | Distinguish "failed" from "ran out of budget". |
| 11 | B | Knapsack comparison used one instance per size, with DP timed as a median of 3 but B&B/ILP timed once | Self-review of the protocol | 5 instances per configuration, same timing protocol for all methods | One instance proves nothing about a method family. |
| 12 | B | Annealing comparison across penalty values used a fixed temperature schedule, confounding "large P is hard" with "schedule is mis-scaled" | Reviewing the SA-vs-P curve | Added an energy-scaled schedule and a swap-move SA control experiment | Build the control that can falsify your explanation. |

