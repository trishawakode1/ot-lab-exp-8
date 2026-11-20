# Track B prompt log (Claude, 2026-11-20)

**Prompts (verbatim from the student):** "ya go ahead, how many things need to be made?" then "Continue".
**AI output:** `knapsack.py`, `scheduling.py`, `qubo.py`, `test_track_b.py`, `run_track_b.py`, then self-review iterations recorded as failure-log entries 9-12.
**Verification note:** 11 unit tests (all exact solvers vs brute force; QUBO<->Ising exact on every bitstring; Max-Cut and penalty encodings; swap-SA keeps cardinality).
**Resulting code change:** TODO add commit link (`[AI:claude] Track B ...`).
**TODO (you):** export the chat; re-run `run_track_b.py`; be able to derive the QUBO->Ising conversion without AI.
