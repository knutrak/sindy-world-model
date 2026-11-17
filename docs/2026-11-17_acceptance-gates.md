# Acceptance gates for the Phase 1 world model (pre-registration)

**Status: FINAL (17 Nov 2026).** Drafted before the final test data was collected; all
thresholds confirmed as written, before any evaluation on the final data. Nothing in this file
changes from here on; the evaluation is run once and reported as it comes out.

## Why

Every model so far was chosen while looking at the validation battles, and `sindyc_07` has also
used the `cf_*` sets as a test set. To make the final claims honest, the criteria for "a valid
model" are written down here first, and then applied **once** to battles nobody has looked at.

## The final test data (collected 17 Nov, not looked at)

From committed code (`allow_dirty=False`), with seeds used nowhere else:

| set | what | seeds | battles |
|---|---|---|---|
| `exp06_intents/final` | intents switch every 1-4 s, like `val` | 8,000,000 + | 1000 |
| `exp06_intents/final_cf_<intent>` | counterfactual branches, like `cf_*` | 8,500,000 + | 6 x 600 |
| `exp07_stalkers/final` | stalkers, like exp07 `val` | 7,800,000 + | 300 |

Rule: until the final evaluation, no script or notebook reads these sets except to check that the
collection is complete and clean (manifests only).

## Candidate models (all fixed before the evaluation; no refitting on final data)

| | model | file / recipe |
|---|---|---|
| A | `model_derived`: the core model, rate-fitted STLSQ | `scripts/build_models.py` |
| W | A's library and thresholds, weak-form fit (2 s windows) | `fit_weak(A, weak_rows(train, A.P, A.names, radius=16))`, `notebooks/weak_sindy.py` |
| B | A's terms, coefficients fitted on rollout error | `sindyc_07_models.pkl`, written by running `sindyc_07` |
| S | B plus the share in reach as a state (11 terms) | the same file |

Baselines (for G5): persistence ("nothing changes"); Lanchester (`H_B' = -a H_R`, `H_R' = -b H_B`,
least squares on the training battles); the physics library without control (`ctrl=False`).

## Shared definitions

- **Start points**: every 8th step of each battle after the first 8 (`make_batch`), with the real
  intents over the horizon. Rollouts start from the measured state, formation held, measured reach
  as the start offset (as in `sindyc_07`).
- **Health error at h**: mean |predicted − real| health over both sides, at h seconds.
- **Margin R² at h**: R² of the predicted against the real *change* in (Blue − Red) health over h.
- **Intervals**: 95%, 1000 bootstrap resamples **by battle** (for `final_cf`, by seed, so the six
  branches of a battle stay together). Comparisons are **paired** (same resample for both models).

## The gates

| gate | question | criterion | applies to |
|---|---|---|---|
| G1 sparsity | Is it readable? | ≤ 3 terms per health equation; ≤ 6 in the distance equation; ≤ 2 alive terms per side; S's share equations ≤ 7 terms per side | all |
| G2 physical reading | Does every term mean something? | every kept term has the sign in the table below | all |
| G3 robustness | Are the terms real? | (a) in 30 bootstrap refits on the training battles (resampled by battle), every kept term is kept in ≥ 90% of refits and its coefficient's 95% range stays on one side of 0; (b) rate-fitted models only: the same term set under derivative estimates SG 5, 9, 17 and weak form 1 and 2 s, health coefficients within ±10% of the median | all (b: A, W) |
| G4 stability | Does it behave in rollouts? | every rollout up to 8 s is finite; health, alive, distance ≥ 0; alive never above its start value | all |
| G5 accuracy | Does it predict? | on `final`: health error at 4 s lower than each baseline, paired interval of the difference entirely below 0 | all |
| G6 horizon | How far does it reach? | **reported, not pass/fail**: the reliable horizon is the largest h in {1, 2, 3, 4, 6, 8} s with margin R² ≥ 0.3 and the lower interval bound > 0, on `final` and on `final_cf` | all |
| G7 no overfit | Was it tuned to the validation battles? | health error at 4 s on `final` ≤ 1.1 × on `val` | all |
| G8 transfer (RQ4) | Does it carry to another unit type? | on exp07 `final`, scaled to stalkers without refitting (`switch_library.scaled`; S's share rates scaled with reach / speed): health error at 4 s lower than persistence and than the scaled no-control model, paired interval below 0 | all |

**Expected signs (G2):**

| term | expected | reading |
|---|---|---|
| `Rr` in Blue's health, `Rb` in Red's | < 0 | units in reach deal damage |
| `Rb·wd` in Red's health | > 0, smaller than `|Rb|` | withdrawing removes part of Blue's fire |
| closing terms `close·<movement>` in distance | < 0 | moving towards the enemy shortens the distance |
| `flee·wd` in distance | > 0 | withdrawing lengthens it |
| alive exponents p (and p + p·focus) | between 0 and 1 | units die as health is lost, never faster than in proportion |
| S: `sa`, `sk` | ≥ 0 | the share follows / relaxes towards the closure |
| S: `sf_<movement>` | ≥ 0 | in contact, units outside reach move in |
| S: `so_wd` | ≥ 0 | withdrawing empties the reach |
| S: `sd` | ≤ 0 | as units die, the survivors' share rises |

## Verdicts

- **Valid interpretable model**: passes G1, G2, G3, G4, G5 and G7.
- **Valid predictive extension**: passes G4, G5 and G7, and beats A on margin R² at 4 s on `final`
  (paired interval above 0), but fails one of G1-G3. Reported with what it costs in readability.
- **Transfers (RQ4)**: additionally passes G8.
- G6 is reported for every model that passes G4.

## Procedure

1. Confirm or change every above; commit this file with status FINAL.
2. Write the evaluation as one committed script that only *reads* models and data
   (`scripts/evaluate_final.py`), and commit it **before** running it.
3. Run it once. Commit its output table and these numbers into the status note unchanged.
4. Anything learned afterwards is reported as post-hoc, separately from the verdicts.
