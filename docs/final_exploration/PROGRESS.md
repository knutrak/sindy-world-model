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
- Diagnostic on cv fold 0 (scratch script, not in the table): A's winner accuracy falls with the share of
  the battle Blue spends withdrawing. It is 0.73 for battles with no withdraw and 0.59 for those
  withdrawing over 35% of the time. No-control stays at 0.68-0.78. No-control also predicts a Blue win
  more often (69% against the true 59%). So its edge is partly a bias that happens to suit this data,
  and partly that A gets whole battles with long withdrawals wrong. Open question: is it the distance
  under withdraw, or the 88% fire reduction?
- Over whole battles, S is no better than A or B (nRMSE 0.137 vs 0.130), even though it is best at
  1-8 s. The duration error is about 3 s for every model, on battles of about 9 s.
- Held-out size (fit ≤ 25 units, score 26-33): A degrades the most (health 4 s 0.075, margin R²
  −0.03). B and S hold up (0.060 / 0.056; 0.29 / 0.39).

## Axes A/B screen: per-unit features, one-step oracle (`screen_AB.py`, 18 Nov)

Held-out R² gain of the SG-9 rate when one *measured* feature is added to A's selected terms (5 folds,
mean ± std). This is an upper bound: in a rollout the feature would need its own equation.

| equation | base R² | best additions (gain) | nothing (< 0.005) |
|---|---|---|---|
| health_blue | 0.734 | enemy volleys smoothed over 5 steps +0.083 (±0.002); enemy overkill +0.022; own nearest-enemy distance × R +0.013; own front-half health +0.010 | dispersion, hull, clusters, velocity, Gini, engaged health, F added to R |
| health_red | 0.742 | V5 +0.084; nearest-enemy × R +0.025; overkill +0.024; front-half health +0.012 | same |
| alive_blue | 0.066 | **own units about to die × enemy V5 +0.103 (±0.002)**; × F +0.089; enemy overkill +0.077 | Gini, min health |
| alive_red | 0.159 | overkill +0.064; **dying × V5 +0.063**; dying × F +0.049 | Gini, min health |

Readings, hedged:
- Health: most of the gain is the **volley rhythm** (V5). `sindyc_08` showed it can't be forecast past
  about 1 s, and that it is worth at most about 0.007 of health error at any horizon. The spatial
  features (dispersion, hull, clusters, relative velocity) add nothing given R. So R already holds
  what geometry has to say about damage *at the same moment*.
- "F replaces R" *loses* 0.17 R² here, against `sindyc_06`'s finding that F matches the game rule. The
  difference is that `sindyc_06` smoothed F like the rates, and this screen uses raw F. So it's not a
  contradiction, but don't read anything into this row.
- **The death law is where the per-unit data has something to add.** Deaths track "units at one
  shot from death × enemy volleys" much better than A's `p (ln H)'`. Its R² goes from 0.07 to 0.17
  (Blue) and from 0.16 to 0.22 (Red). That is the game's mechanism: a unit dies when a shot lands on a
  unit with ≤ 6 hp. A rollout would need the count of units about to die, so the next step is to check
  whether that count follows from (H, A) through a closure. If it does, test it as death law G/hazard.
- Overkill helps both equations a little. It needs a target-allocation model, so it's out of scope.

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
