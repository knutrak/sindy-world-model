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

## Phase 1 reference rows (harness, 18 Nov)

Every model was refitted per split. cv = mean over 5 folds of `train` (fold std in brackets for the
key columns). Whole-battle metrics run one rollout per battle from step 0 to the end under the real
intents. They are **new**: the gates only scored 1-8 s horizons.

| cv | health 4 s | margin R² 4 s | battle nRMSE | winner acc. | duration err. (s) | terms |
|---|---|---|---|---|---|---|
| persistence | 0.225 | −0.04 | 0.564 | 0.623 | 49.3 (never ends) | 0 |
| Lanchester square | 0.099 | −0.35 | 0.185 | 0.646 | 4.5 | 2 |
| Lanchester mixed | 0.099 | −1.51 | 0.204 | 0.631 | 3.8 | 4 |
| no control | 0.077 | 0.01 | 0.191 | **0.737 (±0.013)** | 3.9 | ~14 |
| A | 0.053 | 0.01 (±0.06) | 0.130 | 0.666 (±0.017) | 3.4 | 11 |
| B | 0.050 | 0.29 (±0.02) | 0.129 | 0.688 (±0.018) | 3.1 | 11 |
| S | 0.048 | 0.41 (±0.02) | 0.137 | 0.688 (±0.024) | 2.8 | 22 |

- The harness agrees with the gate numbers. Val health error at 4 s is 0.055 / 0.052 / 0.050 for
  A / B / S, against 0.0551 / 0.0518 / 0.0502 in the status note. The stalker transfer gives 0.040 /
  0.031 / 0.028.
- **New, and worth a look:** from step 0, the *no-control* model calls the winner more often
  (0.74) than A, B or S (0.67-0.69), although it is worse at every horizon metric. The same holds on
  val, size and stalkers. Winner accuracy for A is barely above "whoever has more health now" (0.62).
- Over whole battles, S is no better than A or B (nRMSE 0.137 vs 0.130), even though it is best at
  1-8 s. The duration error is about 3 s for every model, on battles of about 9 s.
- Held-out size (fit ≤ 25 units, score 26-33): A degrades the most (health 4 s 0.075, margin R²
  −0.03). B and S hold up (0.060 / 0.056; 0.29 / 0.39).

## Log

- **18 Nov.** Harness, reference models and tests committed (`5fcd2ec`); reference rows run
  (64 jobs, none failed). Literature scan in `01_literature.md`.
- **18 Nov.** Phase 0 audit written (`00_audit.md`, `9dcda0d`). `test2` collection scripts committed
  (`9135ad0`), collection running (logs: `data/logs/test2_2026-11-18/`).
- **Housekeeping, 18 Nov.** `src/sindy_wm/models/trajectories.py` had a staged change (a
  `danielson` argument) that the working tree reverted. That made the tree dirty and blocked the
  clean collection. It is kept in `git stash` ("trajectories.py: staged danielson param …"). To get
  it back: `git stash apply --index`.

## Next

Queued, in order: per-unit features → E/G (thresholds, ensemble, death laws) → D/I/J
(size-scaled library, MLP) → C/E (optimizers, symmetry, phases). Then the A/B feature screens.
