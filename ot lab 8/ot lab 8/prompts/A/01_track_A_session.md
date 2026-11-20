# Track A prompt log (Claude, 2026-11-15)

**Original prompts (verbatim from the student):**
1. Uploaded the assignment PDF + "pls do this submission for me"
2. "yes get sounds get u can get started"

**AI response summary:** Claude stated what it could and could not do (cannot publish to GitHub, record the demo, run xv6/real quantum hardware,
or fabricate logs), proposed a skeleton + Track A first, then wrote `optimizers.py`, `problems.py`, `test_track_a.py`, `run_track_a.py`.

**Revised prompts / self-review iterations (Claude's own critical review, see `failure_log.md` #3-#7):** truncated lr grids, unfair untuned baselines,
misleading plot and label were detected and corrected before results were reported.

**Resulting code change:** TODO — add commit link after `git commit -m "[AI:claude] Track A ..."`.

**Verification note:** 9 unit tests (finite-difference check of backprop, HVP vs exact Hessian, GD rate vs (κ-1)/(κ+1) theory,
Newton exact in one step on a quadratic, Gauss–Newton parameter recovery). All pass.

**TODO (you):** export the claude.ai chat (or screenshots) into this folder, remove any secrets.
