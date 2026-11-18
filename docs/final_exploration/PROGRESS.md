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
- **Follow-up (scratch screen, same folds):** the hazard law doesn't survive a closure. "About to die"
  predicted from (H, A) through a binned function of mean health per unit only reaches R² 0.26-0.31.
  With R in place of measured volleys, it explains nothing (alive R² −0.01 / −0.13). The rhythm, not
  the count, carried the gain. **What does help** is A's own law written with the regressor the
  rollout uses. In a rollout H' = k R_enemy, so A' = p k R_enemy A / H. Regressing the alive rate on
  R_enemy·A/H, instead of dividing by the noisy smoothed (ln H)', gives held-out alive R² 0.20 / 0.25
  (A's fit: 0.07 / 0.13). That is the errors-in-variables bias `sindyc_06` found, removed within
  one-step SINDy. On val: p_B 0.52 → 0.66, p_R 0.51 → 0.54, Red's focus term 0.14 → 0.24; alive error
  at 4 s 0.98 → 0.92; health 0.055 → 0.053. Now in the harness as **G/alive-reach**.
- Overkill helps both equations a little. It needs a target-allocation model, so it's out of scope.

## Axes E/G screen (`explore_EG.py`, 88 jobs, 18 Nov)

cv means over 5 folds. The fold std is about 0.0005 for health at 4 s, 0.06 for margin R² at 4 s,
0.011 for alive at 4 s and 0.017 for winner accuracy. Full table: `results_summary.md`.

| cv | health 4 s | mR² 4 s | alive 4 s | battle nRMSE | winner | blow-up | terms |
|---|---|---|---|---|---|---|---|
| A (thr 0.15) | 0.053 | 0.01 | 0.959 | 0.130 | 0.666 | 0 | 11 |
| thr 0.02 / 0.05 | 0.052 | 0.17 / 0.14 | 0.95 | 0.121 / 0.122 | 0.67 / 0.68 | **2.6% / 2.2%** | 19 / 16 |
| thr 0.1 / 0.2 | 0.053 | 0.03 / 0.01 | 0.96 | 0.132 / 0.130 | 0.66 / 0.67 | 0 | 12 / 11 |
| thr 0.3 | 0.060 | −0.06 | 1.026 | 0.141 | 0.698 | 0 | 10 |
| E-SINDy (30 bags, keep ≥ 0.5) | 0.053 | 0.01 | 0.960 | 0.130 | 0.666 (Brier 0.331) | 0 | 11 |
| alive: no intent terms | 0.053 | 0.01 | 0.970 | 0.130 | 0.666 | 0 | 10 |
| alive: proportional (p = 1) | 0.052 | 0.02 | 0.966 | 0.136 | 0.678 | 0 | 10 |
| **alive: reach regressor** | **0.051** | 0.02 | **0.905** | **0.127** | 0.674 | 0 | 11 |
| alive: rollout-fitted p | 0.051 | 0.02 | 0.900 | 0.128 | 0.677 | 0 | 11 |
| B / S | 0.050 / 0.048 | 0.29 / 0.41 | 0.84 / 0.81 | 0.129 / 0.137 | 0.69 / 0.69 | 0 | 11 / 22 |

- **E-SINDy inclusion probabilities** (30 bootstrap refits by battle, every fold): all 11 of A's terms
  are kept in **100%** of refits. The alive intent terms that made A fail G3b come in rarely: Blue
  focus 10%, Blue hold 13% (fold-dependent, 0-30%), Red withdraw 3%. Red focus is kept in 100%.
  So the term set is stable under resampling and only moves with the *derivative estimate* (G3b).
  That is a statement about the alive equation's sensitivity to smoothing, not to the data.
- The ensemble's win probability barely calibrates better than A's 0/1 call (Brier 0.331 vs 0.334).
  The members barely differ, because the uncertainty is in the closure, not in the coefficients.
- **The reach-regressor alive law** matches the rollout-fitted alive law (alive 4 s 0.905 vs 0.900;
  A 0.959, about 5 fold-stds better) without any rollout fitting. It improves health and battle
  nRMSE a little. On held-out size it helps (health 4 s 0.070 vs 0.075). On **stalkers** its alive error
  is worse (0.651 vs 0.599), so the scaled transfer prefers A's exponents.
- Lower thresholds buy margin R² with extra, partly cancelling terms and 2-3% blow-ups. That is the
  same trade-off as `sindyc_02`.
- **Winner accuracy keeps moving against the other metrics** (thr 0.3 is the worst model and has the
  best winner score, 0.698). With close battles and a 59% Blue win rate, it rewards a bias towards
  Blue more than fidelity. I'll treat it as a secondary metric and report Brier only next to it.

## Axes D/I and J screen (`explore_DIJ.py`, 18 Nov)

| | health 4 s (cv / size) | mR² 4 s (cv / size) | mR² 8 s (cv) | alive 4 s (cv) | battle nRMSE (cv) | parameters |
|---|---|---|---|---|---|---|
| A | 0.053 / 0.075 | 0.01 / −0.03 | −0.83 | 0.959 | 0.130 | 11 terms |
| size-scaled, thr 0.15 | identical to A: no size term survives | | | | | 11 |
| size-scaled, thr 0.05 | 0.052 / 0.074 | 0.13 / 0.16 | −0.75 | 0.952 | 0.121 (2.2% blow-ups) | 17 |
| S | 0.048 / 0.056 | 0.41 / 0.39 | −0.09 | 0.810 | 0.137 | 22 |
| MLP (64, 2 s rollout loss) | 0.036 / 0.043 | 0.56 / 0.61 | −0.37 | 0.729 | 0.111 | 5,509 weights |
| MLP wide (128, 4 s) | **0.033 / 0.040** | **0.62 / 0.65** | **0.17** | **0.647** | **0.092** | 19,205 weights |

- **Battle size (D/I):** at A's threshold, STLSQ keeps none of the size-dependent copies of the
  terms. At 0.05 it keeps exactly one in every fold: Blue's damage taken per Red unit in reach,
  Rr × z, +0.012 per 10 units (about 7% less per unit in larger battles). It doesn't help on the
  held-out larger battles. Reading: the coefficients are nearly size-free, so one model covers 11-33
  units with no size parameter. A's size degradation comes from elsewhere, plausibly the closure:
  B and S hold up on `size` with the same terms.
- **Cost of interpretability (J):** the black-box world model on *the same* inputs (team state,
  formation at the start, intents, start share in reach) is the new upper reference. It is clearly
  better on every horizon metric and over whole battles, and it holds on larger battles. In health
  error at 4 s, S closes about 25% of the gap from A to the wide MLP (0.053 → 0.048 → 0.033). In
  margin R² it closes about 65% (0.01 → 0.41 → 0.62). This fits `sindyc_07`'s tree ceiling (0.64-0.70
  from the start state) and confirms on a dynamic model that the team-level state is not the limit.
- The MLP's winner accuracy (0.61-0.65) is *below* A's, although its trajectories are better. That is
  the third sign that winner accuracy from step 0 doesn't measure fidelity here.

## Axes C and E screen (`explore_CE.py`, 72 jobs, 18 Nov)

cv differences from A, paired by fold (mean; folds better out of 5):

| | health 4 s | mR² 4 s | alive 4 s | battle nRMSE | terms | reading |
|---|---|---|---|---|---|---|
| SR3 (L0, λ matched to thr 0.15) | −0.0024 (5/5) | +0.087 (5/5) | +0.071 (0/5) | −0.001 (4/5) | 10 | gain is **shrinkage**: health coefficients 0.148 / 0.156 against 0.177 / 0.186, about B's 0.156 / 0.165; drops `flee·wd` |
| SR3 unbiased | +0.0033 (0/5) | −0.112 (0/5) | +0.026 (0/5) | +0.003 (0/5) | 10 | A's coefficients without `flee·wd`: worse everywhere |
| SSR κ 0.01 / 0.1 | +0.010 / +0.004 | −0.04 / +0.07 | worse | +0.047 / +0.033 | 15 / 21 | more terms, 2-4% blow-ups |
| FROLS κ 0.01 / 0.1 | +0.014 / +0.17 | worse | worse | worse | 9 / 4 | drops `Rb·wd`; at 0.1 degenerate |
| **symmetric** (one k for both sides, one p) | −0.0002 (4/5) | +0.015 (4/5) | −0.011 (5/5) | −0.0026 (5/5) | 9 | k = −0.182 (A: −0.177 / −0.186); fewer numbers, ties or slightly better |
| symmetric health only | same | same | same | same | 10 | |
| phase hybrid (approach / engaged / mop-up) | −0.0017 (5/5) | −0.046 (0/5) | −0.025 (5/5) | +0.0025 (0/5) | 38 | **5.5% blow-ups**; approach-phase alive law unphysical (p_wd ≈ 1.1) |

- None of the optimizers beats STLSQ on A's library once shrinkage is separated out. The SR3 /
  B / "fitting bias" story is one effect: smaller firepower coefficients compensate for the reach
  closure over-predicting contact.
- Side symmetry is consistent with identical units and costs nothing. It is cheap evidence for the
  physical reading of k.
- An explicit phase switch adds parameters and instability, with no rollout gain over the smooth gates.

## Depth: S's share equation by one-step SINDy (val, scratch run, 18 Nov)

STLSQ on the SG-9 rate of the measured share, with S's seven candidates per side, at thresholds 0.05
down to 0.003, with and without S's sign rules:
- At 0.05 it keeps **only** `sa` (share′ ≈ 0.97 closure′; one-step R² 0.78 / 0.71). At lower
  thresholds the relaxation term `sk` comes in with the **wrong sign** (−0.066), and the sign rule
  removes it.
- Rollouts never get better than A: margin R² at 4 s 0.03-0.075 (S: 0.41), health 0.055-0.056.
- **Reading (hedged):** S's gain lives in the slow relaxation of the reach offset. One-step rates
  can't identify it, because the fast closure rate dominates the derivative. It only appears when
  fitting on trajectories. That is the strongest evidence so far for "trajectory refinement matters
  here". Queued on all splits as `deep/share-sindy` for the record.

## Depth stage (`explore_deep.py`, 32 jobs, 18 Nov)

- **A\*** = A + one firepower coefficient for both sides + the alive law fitted on R_enemy·A/H.
  One-step fits only, 10 terms. Paired against A on cv, it is better in 5/5 folds on health 4 s
  (−0.0019 ± 0.0002), alive 4 s (−0.055 ± 0.008) and battle nRMSE (−0.004 ± 0.001); margin R² ties.
  On held-out size: health 4 s 0.070 vs 0.075. On stalkers, margin R² 4 s is *worse* (0.12 vs 0.21) and
  alive is worse (0.65 vs 0.60).
- **B\* and S\* (rollout refit started from A\*) end at exactly B's and S's coefficients** (fold
  differences ~1e-5). Trajectory refinement has one optimum here, independent of the start. So B and
  S go to Phase 3 as they are.
- **One-step share SINDy on all splits:** margin R² 4 s 0.07 (cv) against S's 0.41. Confirmed negative.
- **A\* coefficient intervals** (30 bootstrap refits by battle, closures fixed;
  `final_Astar_bootstrap.csv`): k = −0.1819 [−0.1826, −0.1812], k_wd = +0.159 [0.158, 0.160],
  p_B = 0.645 [0.642, 0.649], p_R = 0.552 [0.545, 0.557], p_R,focus = 0.244 [0.235, 0.254]. All
  terms are kept in 30/30. These intervals cover data noise only, not the closure or the derivative
  estimate.
- **A\* against the derivative estimate (G3b-style, SG 5/9/17/33, scratch run):**

  | | SG 5 | SG 9 | SG 17 | SG 33 |
  |---|---|---|---|---|
  | A: k_B / k_R | 0.186 / 0.193 | 0.177 / 0.186 | 0.174 / 0.183 | 0.137 / 0.139 (+ Ar, Ab terms) |
  | A\*: k | 0.190 | 0.182 | 0.179 | 0.175 |
  | A: p_B / p_R | 0.47 / 0.45 | 0.52 / 0.51 | 0.61 / 0.60 (+ wd) | 0.63 / 0.61 (+ focus_B) |
  | A\*: p_B / p_R | 0.63 / 0.52 | 0.64 / 0.55 | 0.66 / 0.58 | 0.67 / 0.60 |

  A\*'s term set is the same at every window, and k stays within ±4% of SG 9. Its alive exponents
  move about 5% (p_B) and 10% (p_R), against A's 25-35%. Caveat: the alive law has no selection
  step, so "same term set" is partly by design. The weak-form estimates of G3b were not re-run.

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
