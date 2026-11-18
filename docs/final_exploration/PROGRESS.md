# Final exploration: progress

Branch `final-exploration`. Newest entries at the top of each section.

## Decisions (18 Nov)

| question | decision |
|---|---|
| Phase 3 test data | **Fresh sets with new seeds** (`test2`): exp06 `test2` 1000 battles (9 000 000+), `test2_cf_<intent>` 6 × 600 (9 500 000+), exp07 `test2` 300 (7 900 000+). Collection script committed before the run (`9135ad0`). Not read until Phase 3. `final` stays the 17 Nov pre-registered test; anything on it from now on is post-hoc. |
| Scope | Harness and reference rows first, then **4-5 focused axes**: new per-unit states and groups (A, B); optimizers / ensembles / constraints, aimed at the alive equation's robustness (E, G); parametric library across battle sizes (D, I); black-box reference (J). The other axes get a short screen or a written reason to skip. |
| Report | Full report as in the brief: ranking, recommended model, draft thesis paragraphs. |
| Dependencies | Allowed: ruptures, hmmlearn, cvxpy, torch. Not installed: PySR. |
| Screening data | A **screening split carved from `train` by battle**. `val` has already been used for every choice since exp06, so it serves as a second check. |

## Log

- **18 Nov.** Phase 0 audit written (`00_audit.md`, `9dcda0d`). `test2` collection scripts committed
  (`9135ad0`), collection running (logs: `data/logs/test2_2026-11-18/`).
- **Housekeeping, 18 Nov.** `src/sindy_wm/models/trajectories.py` had a staged change (a
  `danielson` argument) that the working tree reverted. That made the tree dirty and blocked the
  clean collection. It is kept in `git stash` ("trajectories.py: staged danielson param …"). To get
  it back: `git stash apply --index`.

## Next

1. Harness (`src/sindy_wm/evaluation/harness.py`): splits, metrics, results table, Pareto plot.
2. Reference rows: A, B, S, Lanchester, persistence, no-control.
