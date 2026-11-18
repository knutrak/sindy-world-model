# Phase 3 plan: test2 evaluation (pre-registration, 18 Nov 2026)

Written and committed **before** the test2 sets are read. The script `experiments/final_exploration/phase3_test.py`
is committed in the same commit and runs **once**. Anything learned afterwards is post-hoc.

## Data

The `test2` sets were collected on 18 Nov from commit `9135ad0` with new seeds, and nobody has
looked at them. Only the manifests were checked: complete, clean.

| split | set | battles | bootstrap unit |
|---|---|---|---|
| test2 | exp06 `test2` (9 000 000+) | 1000 | battle |
| test2_cf | exp06 `test2_cf_<intent>` (9 500 000+) | 6 × 600 | seed (six branches together) |
| test2_stalkers | exp07 `test2` (7 900 000+), marine models scaled without refitting | 300 | battle |

Every model is fitted on all of exp06 `train`, as in the `val` split. Nothing is fitted on test2.

## Candidates (from the validation results, three points on the Pareto front)

| candidate | what | terms / numbers | why chosen |
|---|---|---|---|
| **A\*** (`deep/A*`) | A + one firepower coefficient for both sides + the alive law fitted on R_enemy·A/H. One-step fits only | 10 / 44 | best one-step model: better than A in 5/5 folds on health 4 s, alive 4 s, battle nRMSE |
| **B** (`ref/B`) | A's support, coefficients refitted on 1-8 s rollouts | 11 / 47 | trajectory refinement. Refitting from A\* gives identical coefficients (B\* = B) |
| **S** (`ref/S`) | B + the share in reach as a state | 22 / 58 | best SINDy model at 1-8 s (S\* = S) |

References: A (`model_derived` recipe), persistence, Lanchester square, no-control, and the MLP world
model (`J/mlp-wide`, 19 205 weights) as the black-box upper reference.

## Metrics

All are harness metrics: health error, margin R², alive and distance error at 1/2/4/8 s; one-step
R²; whole-battle nRMSE, winner accuracy, Brier, final-health error, duration error, blow-up rate.
Paired 95% bootstrap intervals (1000 resamples) of *candidate − A* for health 4 s, margin R² 4 s
and battle nRMSE.

## Expectations, stated before running (from cv / val)

1. A\* beats A on health 4 s and battle nRMSE on `test2` (paired interval below 0). On margin R² 4 s
   it ties.
2. B and S beat A on margin R² 4 s on `test2` and `test2_cf`, which would replicate 17 Nov. S does
   *not* beat A or B on whole-battle nRMSE.
3. The MLP is better than every SINDy model on health 4 s, margin R² 4 s and battle nRMSE.
4. On `test2_stalkers`, A\* is *not* better than A on margin R² 4 s (screening: 0.12 vs 0.21). B and S
   transfer better than A.
5. Winner accuracy is reported but not used to rank. In screening it moved against every fidelity
   metric.

Each expectation is reported as met or not met, whatever the outcome.
