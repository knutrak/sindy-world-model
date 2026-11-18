# Final exploration, Phase 0: audit (18 Nov 2026)

Read-only audit of what exists before the final exploration. Nothing in the pipeline was changed.
Sources: `docs/2026-11-11_status-and-next.md`, `docs/2026-11-17_acceptance-gates.md`,
`docs/2026-11-17_final-evaluation.md`, `scripts/build_models.py`, `scripts/evaluate_final.py`,
`src/sindy_wm/`, the raw manifests and the processed tables.

## 0. Points to settle first

1. **The held-out test data is already spent.** The gates were pre-registered (`900db91`) and applied
   once to exp06 `final`, `final_cf_*` and exp07 `final` (`30420c9`). Any model chosen by this
   exploration and then scored on those sets is a post-hoc result, not a test. A clean Phase 3
   test needs **new battles with new seeds** (see section 6). Collection is cheap: 1000 exp06
   battles took about 6 minutes.
2. **The brief describes an older stage of the project.** It mentions unit counts as inputs, centroid
   distance, sigmoid gating, Lanchester and naive baselines, and winner accuracy. That matches the
   October Level A runs (`runs/`). The current models (A, W, B, S) are SINDYc with Blue's intents as
   control, a reach closure, an alive rule and rollout-fitted variants. They are scored on horizon
   errors from start points, not on whole battles. Several directions in the brief have already
   been run (section 5). A harness that gives the October metrics would not be comparable with the
   gate results without a bridge.
3. **The exploration's scope works against the plan in the status note.** Worry 2 there is "too many
   branches", and the note says to leave phases, groups and terrain until the core is settled. The
   memory note from today says you lean towards A as the core model. A broad search now creates more
   forking paths after a pre-registered evaluation, and the thesis would have to report it that way.

## 1. Data

All raw data has per-unit snapshots. Everything from exp03 onwards logs the full field set.

| experiment | scenario | Blue policy | step_mul | battles | per-unit fields |
|---|---|---|---|---|---|
| exp01 | 5m_vs_5m | AttackNearest, aggression 0.5-1.0 | 1, 2, 4, 8 | 16 × 500 | x, y, hp, shield, cooldown, alive (no velocity / target) |
| exp02 | 5m_vs_5m, random spawn | as exp01 | 1, 2, 4, 8 | 16 × 100 | as exp01 |
| exp03 | 10m_vs_10m, random spawn | as exp01 | 1, 2, 4, 8 | 16 × 500 | full (adds vx, vy, target_team, target_id, target_in_range) |
| exp04 | 10m, sampled starts | SwitchingPolicy | 2 | 300 (pilot) | full |
| exp05 | 10m, sampled starts | SwitchingPolicy | 2 | 20 000 + 5 000 held-out | full |
| **exp06** | 10m_vs_10m map, `CloseBattleSampler`: Blue 5-19, Red 6-14 marines, force ratio 0.75-1.33, start health 0.6-1.0, spawn distance 4-16 | `IntentSchedule`: advance / hold / withdraw × nearest / focus, switching every 1-4 s | 2 (0.125 s per step) | train 6 × 1000 (seeds 0-5999), val 1000 (3 000 000+), cf 6 × 600 (4 000 000+), cftrain 6 × 2000 (5 000 000+), pilot 6 × 300 (2 000 000+), **final 1000 (8 000 000+), final_cf 6 × 600** | full |
| **exp07** | 10s_vs_10s stalkers, same sampler | IntentSchedule | 2 | train 1500, val 300, **final 300** | full (shields used) |

- A median exp06 battle lasts 74-76 steps (about 9.3 s). Blue wins 56-60%.
- Red is the built-in sticky-nearest AI. Blue is the only side with a varied policy.
- Processed team-level tables (`data/processed/exp06_intents/*.parquet`, one row per step) contain:
  health, alive, centroid distance, front distance, reach (units within ρ = 5.75 of an enemy),
  formation depth / width, intents, seed, outcome, n_blue / n_red. There are also tables for firing
  counts (train/val), formation numbers and hold-vs-advance pairs.
- Splits are **by battle with disjoint seed ranges**, and every set was collected separately, so no
  battle is shared between splits.
- Note: simulation and analysis share the same Python 3.12 `.venv` (smaclite is a dependency in
  `pyproject.toml`). The brief mentions a separate 3.10 env; `~/Documents/starcraft/.venv` may be that
  env, but this repo doesn't use it.

## 2. Current models and their final numbers

Common scale: health 1 = ten full marines. State x = [H_B, H_R, A_B, A_R, d]. Control u = [adv, hold,
wd, focus]. R = units in reach = A · Φ((ρ − gap)/spread), the derived closure (λ, β, g fitted, ρ = 5.75).

**A (`model_derived`, STLSQ on SG-9 rates, threshold 0.15, 11 terms):**

```
H_B' = −0.177 R_R
H_R' = −0.186 R_B + 0.163 R_B·wd
d'   = −6.008 c·adv − 3.158 c·wd + 0.296 f·wd − 3.183 c·hold·far − 5.915 c·hold·contact
(ln A_B)' = 0.517 (ln H_B)'
(ln A_R)' = (0.509 + 0.136 focus) (ln H_R)'
```
c, f = closing / flee switches (24 numbers fitted beforehand with `curve_fit`).

**B (same terms, coefficients fitted on rollout error at 1-8 s):** kb −0.156, kr −0.165, kr_wd +0.114,
d_adv −7.70, d_wd −1.96, d_flee +0.345, d_hold_far −2.16, d_hold_contact −4.72, p_b 0.766,
p_r 0.677, p_r_focus 0.153, reach-offset decay τ_B 0.417 s.

**S:** B plus the share in reach as a state (5 + 6 terms). **W:** A's library, weak-form fit.

Final, pre-registered evaluation (run once, 17 Nov):

| | A | W | B | S |
|---|---|---|---|---|
| verdict | not valid (G3b, alive term set only) | not valid (G3b) | valid interpretable | valid predictive extension (fails G2) |
| health error 4 s, `final` | 0.0536 | 0.0529 | 0.0499 | 0.0484 |
| margin R² 4 s, `final` | 0.023 [−0.15, 0.16] | −0.028 | 0.275 [0.22, 0.33] | 0.399 [0.36, 0.44] |
| reliable horizon (`final` / `final_cf`) | 2 / 2 s | 2 / 2 s | 3 / 4 s | 4 / 4 s |

On `final`, every model beats persistence, Lanchester and the model without control on health error at
4 s. Earlier references on the test/val sets: the tree ceiling has margin R² 0.70 at 4 s and health
error 0.034, and generic SINDYc with 138 terms has health error 0.053.

The October Level A runs (`runs/*/metrics.json`) are not comparable. They use a different scenario,
policy and metric set (winner accuracy 0.18 from the start, against a majority baseline of 0.82).

## 3. Evaluation protocol and its weak points

**Current protocol** (`intent_model.make_batch`, `horizon_errors`, `scripts/evaluate_final.py`):

- start points every 8th step after step 8
- open-loop rollouts under the real intent sequence, at h ∈ {1, 2, 3, 4, 6, 8} s
- health error and margin R²
- paired bootstrap by battle (by seed for cf)
- closures, rates and standardisation fit on train only (`training_rows()`)

**Weak points:**

1. **Validation is heavily reused.** Every modelling choice since exp06 (thresholds, closures, B/S
   candidates) was made while looking at `val`, and `cf_*` served as a test in `sindyc_07`. A new
   exploration on `val` adds to that. Screening should use **a fresh validation split carved from
   train** (by battle) so that `val` can serve as a second, less-used check.
2. **Formation is held and the start offset uses measured reach.** Rollouts get the true formation and
   the true reach at the start point. That's legitimate as an initial condition, but it isn't a
   from-the-start whole-battle rollout. Long-horizon (> 8 s) and whole-battle metrics (winner, final
   health, duration) are **not** in the current protocol, even though a median battle is ~9 s.
3. **SG smoothing is centred.** It uses future steps within a battle. That's fine for fitting rates,
   but rates must never enter evaluation inputs. They don't today.
4. **Bootstrap intervals cover data noise, not fitting noise.** No repeated fits over seeds/folds
   in the main table, except G3a's 30 refits.
5. **No calibration metric.** Models are deterministic, so there are no win probabilities.
6. **The October `runs/` pipeline** (Level A) has its own split (`per_episode.csv`) and metrics. It is
   unrelated to exp06 and should not be mixed into one table without a note.
7. **Only one map and one Red policy.** "Held-out scenario" options in the existing data: battle size
   (exp06 spans 11-33 units, so you can hold out the largest), force ratio extremes, unit type
   (exp07), and the Blue policy (exp01-03 AttackNearest, different pipeline).

## 4. PySINDy and tooling

Installed: **pysindy 2.1.0**, numpy 2.5.3, scikit-learn 1.9.1, scipy 1.18.1, Python 3.12.13.

| feature | in 2.1.0? | used so far? |
|---|---|---|
| STLSQ | yes | own STLSQ per equation (`intent_model.stlsq`), not `ps.STLSQ` |
| SR3 (L0/L1, trimming) | yes | no |
| SSR, FROLS | yes | no |
| EnsembleOptimizer (bagging, library ensemble) | yes | no (G3a bootstrap is hand-rolled) |
| ConstrainedSR3, TrappingSR3, StableLinearSR3, SINDyPI | yes, once `cvxpy` is installed (*corrected 18 Nov*: hidden without it) | no |
| MIOSR, SBR | need `gurobipy` / `numpyro`; not installed | — |
| WeakPDELibrary | yes | own weak form in `notebooks/weak_sindy.py` |
| SINDyPILibrary | yes | no |
| ParameterizedLibrary, TensoredLibrary, GeneralizedLibrary | yes | no (libraries are built by hand in `intent_model.library`) |
| DiscreteSINDy | yes | October runs only |
| SmoothedFiniteDifference (SG) | yes | yes (window 9; 5-33 swept in `sindyc_08`) |
| SINDyDerivative (TV, kalman, spline via `derivative`) | yes | no |

Not installed: `ruptures`, `hmmlearn` (light), `cvxpy` (medium; would bring back constrained fits),
`gurobipy` (MIOSR needs it and a licence), `torch` (neural ODE / SINDy autoencoder; heavy), `pysr`
(needs Julia; heavy). Linear equality constraints, such as symmetric coefficients, can be done with
scipy alone.

## 5. What the brief asks for that already exists (by axis)

| axis | already done | where |
|---|---|---|
| A states | reach (units within ρ of an enemy), firing count F, formation depth/width, front distance, cooldown "about to fire", share in reach as a state | `sindyc_06`, `sindyc_07`, `sindyc_08` |
| B groups | "in reach" vs not is the closure; no group-level state | — |
| C phases | far/contact split of hold; movement switches; no hybrid/HMM/change-point | `sindyc_04`, `sindyc_05` |
| D library | physics library; generic 138-term SINDYc | `sindyc_02` |
| E optimizers | STLSQ threshold sweep; rollout refit (B) | `sindyc_02`, `sindyc_07` |
| F derivatives | SG 5-33, weak 0.5-2 s, box smoothing over volley period; step_mul data exists (exp01-03) | `sindyc_08` |
| G events | alive rule; attenuation of p; wounded pool | `sindyc_06` |
| I generalisation | stalkers by scaling; battle size | `sindyc_04`, `sindyc_07` §8, `sindyc_08` |
| J reference | persistence, Lanchester, no-control, gradient-boosted tree ceiling | `00_model_overview`, `sindyc_07` |

New ground in the brief: dispersion / hull / nearest-enemy stats, health distribution (Gini, about to
die), engaged-vs-not group states, front/rear, clusters, hybrid / HMM phases, learned gate centres,
SR3 / SSR / FROLS / ensembles, symmetry constraints, ParameterizedLibrary across sizes, SINDy-PI,
hazard death laws, win probability and calibration, PCA / delay embeddings, and an MLP / neural ODE
reference on the same states.

## 6. Per-unit logs, and what would need new data

**Available without simulating again** (exp03-07): positions, velocities, hp, shield, cooldown,
alive, current target and whether it is in range, every 0.125 s. Every state in axes A and B can be
computed from these, and so can "firing" (cooldown reset), overkill (damage to a target already
dead within the step) and per-cluster aggregates.

**Needs new data:**

| need | why | size |
|---|---|---|
| fresh held-out test (exp06-like, new seeds), and optionally a matching stalker set | `final` is spent; Phase 3 needs untouched battles | ~1000 + 600×6 cf + 300 stalkers, about 30-60 min |
| a held-out scenario that isn't just a slice of exp06 (e.g. Blue 20-30 vs Red 20-30, or a terrain preset such as NARROW) | a stronger RQ4 test than slicing by size | ~1000 battles |
| a second Red policy, or Blue policies outside the intent set | generalisation across policies (axis I) | optional |
| mixed unit types (e.g. 3s5z-like) | axis B per-type groups; flagged optional | optional, more code |

None of this has been collected. I'm asking before starting any of it.
