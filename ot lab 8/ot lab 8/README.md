# Assignment 8 — Optimize Anything (work in progress)

Status: **Tracks A and B complete (tested)**. Tracks C–J pending. See `CHECKLIST.md` for syllabus coverage.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd tracks/A_continuous && python -m unittest test_track_a -v && python run_track_a.py && python make_tables.py
```

## AI tools register
See `prompts/ai_tools_register.md` (must be pasted here and kept complete).
Prompt logs: `prompts/<track>/`. Failure log: `prompts/failure_log.md`. Tool comparison: `prompts/tool_comparison.md` (to be completed by the student).

## Honesty statement
All numbers in `results/` are produced by the scripts in this repo. Claims are limited to what the measurements support; see each track's "Limitations".
Real quantum hardware, xv6/QEMU and multi-core measurements must be run by the student on suitable machines and are marked as such.
