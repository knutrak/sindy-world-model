# Status and next steps (first written 11 Nov 2026, fully updated 16 Nov 2026, afternoon results added)

This is the place to start. **New since the meeting: the section "Results 16 Nov (afternoon)"
below.** Some statements in "Where I stand" are revised there; they are marked *(revised)*.

Other places with details:

- `notebooks/00_model_overview.ipynb`: every model side by side (what is SINDy, what is designed,
  what is a game rule), predictions against real battles, horizon errors with bootstrap intervals,
  and the figures for the meeting brief.
- `notebooks/00_findings_summary.ipynb`: ranked findings.
- `notebooks/sindyc_01` … `sindyc_05`: the experiments, in order.
- `notebooks/sindyc_06_reach_as_state.ipynb`: does units in reach need a state? (tests 0.5 and 1-7).
- `notebooks/sindyc_07_what_limits_the_model.ipynb` (helpers in `notebooks/sindyc_07_lab.py`):
  what limits the model, two improvements (B and S), verification, stalker transfer. Fitted
  parameters in `data/processed/exp06_intents/sindyc_07_models.pkl`.
- `scripts/build_firing_counts.py`: units firing per side and step (used by `sindyc_06`).
- `docs/2026-11-10_sindyc-intents-exploration.md`: running log of the SINDYc work.
- `docs/2026-11-16_supervisor-meeting.tex`: meeting brief (figures from section 10 of the
  overview notebook, written to `docs/figures/`, which is git-ignored).
- `docs/findings.md`: F1-F26. **Not updated** with anything from exp06 onwards.

## Where I stand, in short

- **A readable world model with control exists** (`model_derived.pkl`). The state is each team's
  health (0-1 scale, 1 = ten full marines) and the distance between the team centres. The control is
  Blue's intent as on/off switches: advance / hold the line / withdraw, plus focus fire.
  - Health: `H_B' = -0.177 R_R`, `H_R' = -0.186 R_B + 0.163 R_B·wd`. Withdrawing removes 88% of
    Blue's fire. SINDy keeps 1-2 of 10 candidates per equation.
  - Distance: `d' = -6.01 c·adv - 5.92 c·hold·κ - 3.18 c·hold·(1-κ) - 3.16 c·wd + 0.30 f·wd`.
    SINDy picks the coefficients; the switch shapes c and f were fitted beforehand with
    `curve_fit` (24 numbers).
  - Units alive: `(ln A_B)' = 0.517 (ln H_B)'`, `(ln A_R)' = (0.509 + 0.136·focus) (ln H_R)'`.
  - Units in reach: `R = A · Φ((ρ - gap) / spread)`. ρ = 5.75 is the game's reach; the three
    factors λ, β, g are fitted.
- **Accuracy over horizons (with 95% bootstrap intervals, resampled by battle).**
  - Health error at 4 s: 0.055 (0.053-0.058). For comparison: "nothing changes" 0.233,
    Lanchester 0.102, no control 0.079, generic SINDYc with 138 terms 0.053.
  - Units in reach: 1-1.4 units off, mostly because the distance is slightly wrong. *(revised:
    the true distance alone barely helps the margin; the closure's form is the bigger part.)*
  - Margin (Blue − Red): R² of its change is 0.44 at 1 s, 0.07 at 4 s and −0.66 at 8 s. It's
    reliable to about 3 s, and a battle lasts 10.4 s on average (median 9.1 s).
- **The model can't choose between intents.**
  - Counterfactuals: 600 battles, each branched 6 ways at a random moment; 522 states remain
    after filtering. The model's picks win 58%, the same as always choosing the best fixed intent;
    the hindsight best wins 74%.
  - The predicted ranking of intents matches the real ranking with a correlation of only
    0.12-0.13.
  - Withdrawing is predicted best in 5% of states; in reality it is best in about 40%.
  - In the game: the model planner wins 69.8%, always hold/focus 70.2%, and the oracle (a copy
    of the game) 94.8%.
- **Why: the team-level state can't see formation.** Holding wins because Blue meets Red with a
  line while Red arrives as a column (more Blue units in reach early). Damage per unit and deaths
  per health lost are the same for hold and advance. Giving the model the real distance removes
  most of the units-in-reach error; giving it the real formation numbers (depth, width) alone
  doesn't help. *(revised: the team-level state is not the limit. A flexible predictor on the
  same state reaches a margin R² of 0.64 at 4 s; the gap is the reach closure's form.)*
- **Tried and didn't work:** movement derived from the game's walking rule (units behind their own
  front are blocked; R² 0.43 vs 0.73); choosing the reach factors from the health fit (gives an
  unphysical term, and the planner collapses to 11%).
- **Transfer to stalkers:** the scaled marine model gives a 4 s health error of 0.040, against
  0.024 refitted. This is in `sindyc_04`; it was left out of the meeting brief.
- **All exp06 / exp07 data is exploratory.** It was collected with uncommitted code
  (`allow_dirty`), and models were chosen while looking at validation numbers. *(revised 17 Nov:
  re-collected from committed code; see "Re-collection 17 Nov". The data is identical, so every
  number in this note stands. Models were still chosen while looking at validation numbers.)*

## Results 16 Nov (afternoon): units in reach, and what limits the model

All on exp06 / exp07 data (re-collected clean on 17 Nov, identical). Validation = 1000 exp06 battles; **test** = the six `cf_*`
sets (600 battles, six branches each), used for no choice in `sindyc_07`. "Margin R²" = R² of the
change in Blue − Red health over the horizon. "Trees" = gradient-boosted trees, used only as a
benchmark of what an input set allows, never as a model.

### Units in reach R: no state needed for memory (`sindyc_06`)

- **Definition kept: ρ = 5.75** (the game's firing test, centre to centre). The action mask is
  not "can shoot": it uses a fixed 6.0 and exists for Blue only.
- **Red picks up targets only within 5.375 but can fire to 5.75.** That explains 77.5% of Red's
  in-reach units with no target. About half of Red's firing in the 5.375-5.75 band is at a Blue
  unit shooting it (the game's exception). Band firing is a third of Red's fire, but a share of R
  already captures it at team level.
- **Health per *firing* unit matches the game rule.** With units firing F (smoothed like the
  health rates): `H' = −0.220 F_enemy` and `−0.222`, against the rule 10 · 9.8 / 450 = 0.219. With
  F, Red's withdraw term disappears. Reading of today's coefficient: 0.177 ≈ 0.219 × about 81% of
  units in reach firing.
- **No memory in R or F given the true current state.** Trees predicting R from the current state
  gain 0.003-0.004 R² from R 1 s earlier (same for F, and for Blue as a control). A better closure
  shape could gain 0.08 (Blue) and 0.20 (Red).
- **The alive rule was biased by how it was fitted.** Fitted on smoothed rates, `p` is pulled
  towards zero (attenuation): about 0.5 units too many alive at 4 s given the true health.
  Refitted on multi-step error, `p` rises 0.52 → 0.72 and the error falls to 0.55 / 0.52 units,
  near the floor of about 0.47 (discrete deaths). A "wounded pool" term (W = A − H) adds nothing
  once the bias is gone. Fixing alive alone does not improve the margin.

### What limits the model (`sindyc_07`, sections 1-3)

- **Ceiling:** trees from the start state (team state, formation, measured reach, intents) reach
  margin R² 0.64 (val) / 0.70 (test) at 4 s and health error 0.032 / 0.034. `model_derived`:
  0.07 / 0.25 and 0.055 / 0.058.
- **The health and alive equations are fine:** with the true *share* in reach (R / A, which doesn't
  leak who died) the unchanged model reaches the ceiling (0.64).
- **The closure has the right inputs but the wrong form.** Trees on exactly its inputs: R² 0.95
  for both sides; the formula: 0.92 (Blue), 0.84 (Red). Intent and d′ add nothing. The formula
  fails when Blue is scattered (depth > 1.5 or width > 3): 6% of steps, 60% of its squared error
  for Red. Three readable closure variants plateau at about 0.92.
- **True distance alone barely helps the margin** (0.07 → 0.19 at 4 s); a tree closure on the
  model's own distance and formation does more (0.28).

### Two improvements, verified (`sindyc_07`, sections 4-6)

- **B: same terms, coefficients fitted on rollout error** (least squares on health, alive and
  distance at 1-8 s, 20,000 training start points), plus the decay time of the measured-reach
  offset per side. No term added or removed.
- **S: B plus the share in reach as a state**, started at the measured share. Seven candidate terms
  per side (follow the closure, relax to it, move in once in contact, withdraw empties the reach,
  deaths), pruned to 11 of 14. The old decaying offset is (nearly) a special case.

  | test set | health 4 s | margin R² 2 s | margin R² 4 s | margin R² 8 s | reach error 4 s |
  |---|---|---|---|---|---|
  | A `model_derived` | 0.058 | 0.46 | 0.25 | −0.45 | 1.34 units |
  | B | 0.055 | 0.55 | 0.39 | −0.31 | 1.23 |
  | S | 0.054 | 0.60 | 0.49 | −0.05 | 1.16 |
  | ceiling (trees) | 0.034 | 0.74 | 0.70 | 0.46 | |

- **Checks:** paired bootstrap by battle, every margin difference B−A, S−A, S−B excludes zero at
  2, 4 and 8 s on validation and test. Train ≈ validation (no overfitting to the data). A second
  training subsample gives coefficients within 15% and the same score. Intent ranking barely moves
  (rank correlation 0.14 → 0.17, win rate 58% → 60%).
- **Tried and no better than B:** formation (depth, width) equations, as fitted or refitted
  multi-step; refitting the closure's λ, β, g on rollouts; a saturating enemy-front term.
- **Costs:** the health coefficients move *away* from the game rule (0.177 → 0.156); some of S's
  terms have no physical explanation yet (Red's contact terms negative, Red's deaths term −2.52);
  fitting on rollout error is a second fitting method next to STLSQ; the candidates were designed
  by hand.

### Weak-form SINDy: B's gain is compensation, not bias correction (`sindyc_07`, section 9)

- `model_derived` is reproduced exactly by `IntentModel.fit(reach_true=True, thr_health=0.15)`.
- Weak-form SINDy (same library, same thresholds, integrals against test functions instead of
  derivatives; `notebooks/weak_sindy.py`) **keeps the same terms**, moves the alive exponents
  towards B's (0.52 / 0.51 → 0.70 / 0.65; B 0.77 / 0.68), and keeps the health coefficients near
  the rate fit (0.184 / 0.194 / 0.170, against B's 0.156 / 0.165 / 0.114).
- Rollouts get slightly **worse** on the margin: test margin R² at 4 s 0.25 → 0.20 (paired
  interval −0.05 to −0.04); health unchanged. Same on stalkers.
- So B's gain does not come from removing the fitting bias. It comes mostly from the health
  coefficients (recorded stages: alive only 0.073 → 0.080; + health 0.215; + offset decay 0.281),
  which compensate for the closure. Removing one bias exposes the other: in `model_derived` the
  two errors partly cancel.
- **Fully weak form (WG, section 9b):** the 24 switch numbers refitted in weak form too, then
  the same weak SINDy fit. The switches fit the distance windows much better (validation weak-form
  R² advance 0.26 → 0.52, hold 0.22 → 0.61, withdraw −0.24 → 0.23), but distance forecasts barely
  change, and some numbers become less physical (advance closing speed −6 → −16, hold-in-contact
  −6 → −21, switch widths near 0). Test: equal to A at 4 s, slightly better at 8 s (margin R²
  −0.45 → −0.37, interval +0.03 to +0.11). Stalkers: worse than A on the margin. No overall gain.

### The derivatives, and is SINDy viable on them? (`sindyc_08`, 17 Nov)

- Health, alive and reach change in whole steps (60% / 90% / 72% of steps unchanged). The raw
  rate is a **volley rhythm**: cooldown runs in game ticks of 1/16 s, so a marine fires every
  10 ticks = 0.625 s = exactly 5 steps (autocorrelation 0.71 at 5 steps, all battle long). Only
  about 40% of the raw health rate's variance is slower than 1 Hz.
- **Correction:** the effective damage rate is 6 / 0.625 = 9.6 per second, so the health rule on
  the common scale is **0.213**, not 0.219 (`sindyc_06` test 1's 0.220 / 0.222 is 3-4% above it).
- Term selection for health and distance is the same across 7 derivative estimates (SG 5-33,
  weak 0.5-2 s); health coefficients within about ±5% up to 2 s windows. The alive exponent moves
  0.47-0.70 with the estimate: that equation is the fragile one.
- Smoothing over whole volley periods (5-step box) removes the rhythm (4% → 0.01-0.04% of the
  power) but changes coefficients and predictions very little (val margin R² 4 s 0.08 → 0.10).
- **Cooldown as a state:** "units about to fire" explains next-step damage only a little better
  than units in reach (R² 0.27-0.29 vs 0.17-0.18). Volley timing alone accounts for at most about
  0.007 of health error at any horizon: 51% of the model's error at 1 s, 13% at 4 s, 6% at 8 s.
- **Battle size (11-33 units in total, no intervals):** larger battles have fewer unchanged
  health steps (70% → 57%) but the same volley rhythm, and units in reach grow only with the front
  (2.1 → 3.7). Formations spread (Blue width 0.8 → 1.4, scattered 4% → 8% of steps); the closure's
  error grows in units (0.26 → 0.63) but not per unit alive (0.064 → 0.079). Health error at 4 s
  relative to starting health falls slightly (13.4% → 11.6%): a weak sign the team-level
  description holds up as battles grow.

### Transfer to stalkers without refitting (`sindyc_07`, section 8)

Same scaling as `switch_library.scaled`; S's per-second share rates scaled with movement time
(reach / speed), a rule fixed before looking. 300 stalker validation battles:

| | health 4 s | health 8 s | margin R² 4 s | margin R² 8 s |
|---|---|---|---|---|
| A scaled | 0.040 | 0.072 | 0.07 | −0.35 |
| B scaled | 0.031 | 0.053 | 0.38 | 0.09 |
| S scaled | 0.027 | 0.045 | 0.48 | 0.22 |
| S refitted on stalkers | 0.022 | 0.038 | 0.57 | 0.45 |
| ceiling (trees, stalker-trained) | 0.015 | 0.026 | 0.72 | 0.69 |

- S scaled closes about two thirds of A's transfer gap on health at 4 s. B−A, S−A, S−B exclude
  zero at 4 and 8 s; S−B not at 16 s. The scaling rule for the share rates barely matters to 8 s.
- **Where scaling doesn't hold:** alive exponents agree with a stalker refit (0.77 / 0.68 vs
  0.75 / 0.72); firepower roughly (stalkers do about 20-25% less per unit in reach than scaled);
  advance closing speed (−10 vs −25), hold before contact and most share terms disagree.

### Open questions for me (from the notebooks)

- Adopt B or S, or keep `model_derived` for RQ1 and report B / S as "what the same structure
  reaches"? (Arguments for and against S were written out in the chat; not decided.)
- How to rewrite RQ2 now that the ceiling result says the team-level state is not the limit.
- Why R needs no memory given the true state, yet the share as a state helps in rollouts.
- Which of S's terms I can explain, and whether rollout-fitted numbers can still be read as physics.

## Phase 1 status (milestone: end of 2026)

| Step (project description, section 5) | Status |
|---|---|
| 1 Understand the environment | Done (F1-F3) |
| 2 Collect and analyse data | Done. exp06 (intents) and exp07 (stalkers) re-collected from committed code on 17 Nov (commit `d0c2669`), identical to the exploratory data |
| 3 First world model (Level A, SINDy) | Done |
| 4 Refine states, libraries, methods | Largely done: reach curve (derived form), distance switches, control, hold split by contact. 16 Nov: R as a state tested (no memory), share as a state (S) and fitting on rollout error (B) tried, exploratory. No Level B (groups) |
| 5 Validate on unseen data against baselines over horizons | Done, with bootstrap intervals (overview notebook, section 5). 16 Nov: a tree ceiling and an untouched test set (`sindyc_07`) |
| 6 Generalise across scenarios | Started: stalkers by scaling; B and S also transfer (`sindyc_07`, section 8). Terrain is possible in SMAClite (see below), not tried |
| 7 Clean world-model interface | Partly: `IntentModel.simulate`, `state_from_game`, `print(shape=True)` |

| RQ | Status |
|---|---|
| RQ1 interpretable model | **Mostly answered.** 1-2 terms per health equation, each with a physical meaning. Caveat: SINDy chooses within a designed library; the switch shapes are fitted outside SINDy. 16 Nov: health per *firing* unit = the game rule (0.220 vs 0.219); fitting on rates biases coefficients (attenuation) |
| RQ2 state representation | **Partly.** 16 Nov: the team-level state allows margin R² about 0.64-0.70 at 4 s (tree ceiling); the limit is the reach closure's form, not the state. R needs no memory given the true state, but the share as a state helps in rollouts. No group-level state yet |
| RQ3 accuracy over horizons | **Answered** with intervals: health to about 4 s, margin to about 3 s (`model_derived`). 16 Nov: B and S lift test margin R² at 4 s from 0.25 to 0.39 / 0.49 (ceiling 0.70) |
| RQ4 generalisation | **Started.** Marines to stalkers by scaling (five properties changed at once). 16 Nov: B and S scaled transfer better than `model_derived` scaled (health 4 s 0.031 / 0.027 vs 0.040; refit 0.022) |
| RQ5 helps an LLM decide | **Not started, and at risk.** The model's per-intent predictions don't separate the options, so the LLM would get little to choose with |

## Main worries

1. Ownership: there is a lot of material, and not all of it is fully understood yet.
2. Too many branches: phases, MPPI, new intents, terrain, unit-level states.
3. RQ5 is weak, although the project description calls it central.
4. The wording of "SINDy discovered it", given how much of the library is designed.
5. The ceiling of the team-level state (it can't see formation). *16 Nov: the ceiling is higher
   than the model; the closure is the limit.*
6. Data and code aren't thesis-ready (dirty data; tests missing; `findings.md` out of date).
   *16 Nov: two more notebooks on exploratory data; nothing committed.*
7. Narrow scenario: one map, mostly 10 vs 10 marines.
8. Short reliable horizon: about 3 s of a roughly 10 s battle. *16 Nov: about 4 s with S.*
9. *New 16 Nov:* rollout-fitted coefficients (B, S) predict better but read less like physics.

## Ideas from 16 Nov, not started

Each comes with what I already know about it.

- **More data of the same kind will probably not help.** The intervals are already narrow, and the
  generic 138-term model is no better. More *varied* data could help (very different start
  formations; more counterfactual branches). A cheap check first: a **learning curve** (refit on 10 /
  25 / 50 / 100% of the battles and watch the 4 s error and the intent-ranking correlation). Flat ⇒
  the state is the limit, not the data.
- **A lower SINDy threshold.** Already swept (`sindyc_02`, section 4): 26 terms instead of 4
  improves the 4 s error only from 0.058 to 0.052, and the extra control terms come in cancelling
  collinear pairs (e.g. −0.102 Rb·wd / +0.102 Rr·wd). **Not tested:** whether the denser model
  *ranks intents* better. That takes minutes on the counterfactual set.
- **Raw distance in the health library** (like the health test in overview 2b). If SINDy drops it,
  that's evidence the switch shape is right and not just assumed.
- **Formation intents** (get in line, ball, column). They make formation a choice instead of a
  hidden accident. They need formation as a *state* driven by the intent (e.g.
  `width' = k·u_line·(w_line − width)`), because a formation persists after the intent ends. Warning:
  the real width/depth didn't help reach in section 6b, so first check that line vs column shows
  up clearly in those numbers.
- **More target rules** (nearest, weakest = today's focus, healthiest, spread fire) as team-level
  switches, not "focus on red_i". Focus today: the team target is the weakest enemy any Blue unit
  can shoot; a unit that can't reach it shoots the weakest enemy it can reach. In the model, focus
  only shows up in Red's units-alive rule (+0.136), not in health.
- **Lowering dt (step_mul): not worth it.** Damage comes as shots every 0.61 s, and the errors are
  model bias (distance, formation), not resolution.
- **Terrain (RQ4 test).** SMAClite supports blocked/walkable grids only: presets SIMPLE,
  CHECKERBOARD, NARROW, RAVINE, OCTAGON, CORRIDOR, PENTAGON, ALL_GREEN, or a custom `"terrain"`
  grid in the scenario JSON. There is no height and no high ground, and terrain doesn't block
  shooting. Good as a "fit on SIMPLE, test on RAVINE/NARROW" generalisation test; it would expose
  the formation blind spot.

## What to do next, in order (before the meeting; superseded by "What to do next (after 16 Nov afternoon)")

1. **Write down what the supervisors said** (section below). In particular: RQ5's framing, and
   whether to extend the state or consolidate.
2. **Consolidate.** Re-collect exp06/exp07 with committed code; tests for `intents`, `planner`,
   `intent_model`, `switch_library`, `geometry`; add F27+ to `docs/findings.md`; freeze the core
   model (`model_derived`).
3. **Two cheap checks on existing data:** the learning curve, and the intent ranking of the
   threshold-0.05 model. Together they say whether the decision problem is about data, about terms,
   or about the state.
4. **Decide RQ5's shape:**
   - (a) extend the state to groups (fighting / approaching) or formation, so the model can see
     what decides between intents;
   - (b) reframe RQ5, e.g. the model helps the LLM *explain* its choices, with an oracle as an
     upper bound;
   - (c) shorter planning horizons (about 3 s).
5. **Write the preparation thesis** around RQ1-RQ3 (plus RQ4 if transfer is cleaned up).

Leave phases, MPPI, unit-level targets and terrain until 1-4 are settled.

## Decided 16 Nov (after the meeting)

- **RQ5 is out of scope for Phase 1.** The intent-ranking gap stays documented, but it doesn't
  drive the work, so the learning curve and threshold-ranking checks are dropped.
- **The one model extension: units in reach R as a state** instead of the algebraic closure
  R = A·Φ(…). Candidate form per side: `R_B' = inflow − losses`.
  - The inflow needs the units still on their way, `(A_B − R_B)`, times closing (−d′ or c·u). It
    can be negative under withdraw.
  - The losses come from the *enemy's* fire (via `R_R`), so the two R equations are coupled.
  - R is a count, so expect a low per-step R², as for units alive. Judge it over the horizon.
- **Success criterion (RQ2/RQ3, not RQ5):** a lower units-in-reach error than the closure's
  (1-1.4 units), and a reliable horizon beyond about 3 s.
- **Plan, six weeks, alongside an exam in another course:**

  | weeks | work |
  |---|---|
  | 1-2 | consolidate: re-collect exp06/exp07 with committed code, tests, F27+, freeze `model_derived` |
  | 3-4 | R as a state: library, fit, horizon errors against the closure |
  | 5-6 | write the RQ1-RQ3 chapters, plus R as a state |

  If the exam falls in weeks 3-4, write first and run the experiment afterwards.

- **Update, 16 Nov afternoon:** the weeks 3-4 experiment was run early, on the exploratory data
  (see "Results 16 Nov (afternoon)"). The success criterion is met by S on the test set (reach
  error 1.34 → 1.16 units at 4 s; margin R² at 4 s 0.25 → 0.49). It must be re-run after the
  re-collection in weeks 1-2. Not yet decided: whether B or S becomes the core model.

## Re-collection 17 Nov

- exp06 (25 sets) and exp07 (2 sets) re-collected from commit `d0c2669` with
  `allow_dirty=False`: every manifest says complete, `git_dirty = False`. Same seeds as before.
- Rebuilt with the committed pipeline (order in `scripts/build_models.py`): state tables, closures,
  `model_rich/sparse/split/derived`, formation tables, firing counts, hold/advance pairs, planner
  results (main, split, derived).
- **Everything is identical to the exploratory data:** every state table (exp06 and exp07), the
  closures and all four models' coefficients, the formation and firing tables, the pairs (the old
  file had one extra, empty column), and every planner battle's winner. So the game and policy
  code didn't change since the exploratory collection, and all results from `sindyc_01`-`08`
  hold on the clean data without re-running.
- Not rebuilt (no recipe in the repo): `model_healthfit.pkl` and `planner_healthfit.csv`
  (negative result, no notebook reads them), and `pairs_hold_advance_formation.parquet`, which
  `sindyc_05` reads.
- The old exploratory copies were deleted after the comparison (17 Nov). Logs:
  `data/logs/recollect_2026-11-17/`.

## Final test set and acceptance gates (17 Nov)

- Draft pre-registration: `docs/2026-11-17_acceptance-gates.md` (8 gates for models A, W, B, S;
  thresholds marked [confirm] still to be settled).
- Final held-out sets collected from commit `eb6ddb8`, clean and complete: exp06 `final` (1000
  battles), `final_cf_<intent>` (6 x 600), exp07 `final` (300). Seeds used nowhere else. **Not
  looked at**: only the manifests were checked.
- `sindyc_07` re-run on the re-collected data to regenerate B and S (`sindyc_07_models.pkl`); its
  outputs came out byte-identical to the committed notebook.

## What to do next (after 17 Nov)

1. **Decide B / S / neither** (see the open questions above), before freezing the core model.
2. **Finish consolidating:** tests for `intents`, `planner`, `intent_model`, `switch_library`,
   `geometry`; F27+ in `docs/findings.md` (including the 16-17 Nov results); a recipe for
   `pairs_hold_advance_formation.parquet`.
3. **Write the RQ1-RQ3 chapters**, plus RQ4 (transfer holds on the clean data).

## Analysis plan: units in reach R as a state (done 16 Nov, see `sindyc_06` / `sindyc_07`)

Outcome per step: (1) the closure's error has patterns by intent and time since contact and stays
correlated for about 1-2 s, but trees show that past R adds nothing given the true current state;
(2)-(4) the flow split was not needed once (1) showed no memory; (5) replaced by S, the share as a
state, tested over horizons on validation and test; (6) not run. Open question answered: "in
reach" stays at ρ = 5.75. The original plan, for reference:

The raw data (`data/raw/exp06_intents`) has per-unit snapshots, so every unit can be followed in
and out of reach (`states/geometry.py` already computes `reach_blue` / `reach_red` from them). New
notebook: `notebooks/sindyc_06_reach_as_state.ipynb`. Steps 1-3 are data analysis only; if step 1
shows no memory, stop there (that is a result too).

1. **Is a state needed? (closure residuals.)** Measured R minus the closure's `A·Φ(…)`, plotted
   against time since first contact and against intent (hold vs advance at the same distance).
   A systematic pattern means R depends on history, and only history justifies a state.
2. **Split R′ into its flows, per side and per step:** in (enter reach), out_moved (leave reach
   alive), out_died (die while in reach); check that dying outside reach is about 0. Then
   R′ = in − out_moved − out_died exactly. Which flows matter, and when?
3. **Scatter each flow against candidate drivers** (no fitting yet):
   - in: units on their way (A − R), closing (−d′), intent, depth;
   - out_died: the enemy's fire (R_enemy, via the damage rate);
   - out_moved: u_wd·R, the flee switch f.
4. **Fit each flow separately with SINDy** (its own library and threshold). Hoped-for form, e.g.
   `in_B = k·(A_B − R_B)·(−d′)/depth_B`, `died_B = p·0.177·R_R`.
5. **Replace the closure in the model and test over horizons** against `model_derived`:
   units-in-reach error (now 1-1.4), health at 4 s (0.055), margin R² at 1/4/8 s. Success = a
   lower reach error and a reliable horizon beyond about 3 s. R is a count, so judge it over the
   horizon, not by per-step R².
6. **Check the explanation:** does the new model predict that hold beats advance on
   `pairs_hold_advance.parquet`?

Open question before step 1: should "in reach" mean any enemy within ρ = 5.75 (today's
definition), or "can actually shoot" (the action mask)? They differ for blocked units and dead
targets.

## Later: mixed armies (supervisor's wish, case 2)

One state per type per side (H_k, A_k, R_k, and a position per type group, since speeds differ).
From the SMAClite code (`smaclite/env/units/unit.py`, `unit_command.py`):

- Per shot: bonus damage (stalker vs ARMORED +5), then the shield takes the full damage, then
  health takes the damage minus armour (stalker armour 1, marine 0). Shields regenerate only after
  10 s without damage.
- Damage per second for one shooting unit:

  | shooter → target | damage per second |
  |---|---|
  | marine → marine | 9.8 |
  | marine → stalker | 9.8 on shield, 8.2 on health |
  | stalker → marine | 9.7 |
  | stalker → stalker | 13.4 on shield, 12.7 on health |

  Known from the rules, so it can serve as a check on what SINDy finds.
- Red's AI is sticky nearest and blind to type: it shoots the closest enemy (or one shooting it)
  until it dies or leaves range. So fire allocation is geometry: whichever type is in front is
  shot first.
- Start with mixed Blue against pure Red. Existing maps to look at: `3s5z`, `2s3z`.

## From the supervisor meeting (16 Nov), fill in

- Is choosing terms within a designed library enough, or should more be found by SINDy?
- Next step: consolidate this model, or extend the state to groups / formation?
- Other: …
