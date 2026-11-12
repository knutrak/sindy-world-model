# SINDYc with intents: exploratory run (10 Nov 2026)

Done "for fun" in one session; finished 11 Nov (notebook 3, restricted refit, runner). **Exploratory:** data collected with `allow_dirty`, models chosen
on the validation set, nothing reviewed by you yet. Stopped early; see "Not done".

## What exists

| | |
|---|---|
| `src/sindy_wm/envs/intents.py` | `IntentPolicy` (advance / hold / withdraw x nearest / focus), `IntentSchedule` (switching, branching) |
| `src/sindy_wm/envs/episode_sampling.py` | `CloseBattleSampler` (draws the strength ratio) |
| `src/sindy_wm/states/geometry.py` | reach, depth, width from positions; `load_team_states`; `state_from_game` |
| `src/sindy_wm/models/intent_model.py` | the SINDYc model: closures, library, STLSQ per equation, `IntentModel` |
| `src/sindy_wm/envs/planner.py` | `ModelPlannerPolicy`, `OraclePlannerPolicy` (real simulator as the model) |
| `scripts/collect_intents.py`, `build_intent_states.py`, `eval_planners.py` | collection, state tables, planner comparison |
| `notebooks/sindyc_01_intents_and_pilot.ipynb` | intents, replay check, leverage pilot (executed) |
| `notebooks/sindyc_02_world_model.ipynb` | the model and ten experiments around it (executed) |
| `notebooks/sindyc_03_counterfactuals_and_planner.ipynb` | ranking, predictability (team vs unit features), planner (executed) |
| `scripts/run_notebook.py` | executes a percent-format `.py` into an `.ipynb` (no nbconvert needed) |
| `data/raw/exp06_intents/`, `data/processed/exp06_intents/` | pilot, train (6,000), val (1,000), cf (600 x 6), cftrain (2,000 x 6); `planner_main.csv` |

## Results (numbers as measured)

1. **The simulator replays exactly**, and the game can be deep-copied: counterfactuals can be measured.
2. **Leverage pilot:** the intent played for the first 4 s decides the winner in 72% of 300 close
   starting states. Best single intent 55%, best of six in hindsight 80%. Hold loses (19%).
3. **The model** (threshold 0.2, health fitted on measured reach):
   - `(health_blue)' = -0.176 Rr`
   - `(health_red)' = -0.185 Rb + 0.164 Rb*wd` (withdrawing removes 89% of Blue's fire)
   - `(distance)' = -6.0 close*adv - 3.2 close*hold - 3.2 close*wd + 0.3 flee*wd`
   - `(log alive_red)' = [0.49 + 0.15 focus] (log health_red)'` (focus fire kills units sooner)
4. **Prediction, validation battles, real intents applied:** health error at 4 s 0.056 (sparse) /
   0.052 (rich); without control 0.079; persistence 0.22. A generic degree-2 polynomial SINDYc gets
   0.055 with 147 terms: **the physics buys compactness, not accuracy.**
5. **Limit:** with the real reach the error is 0.028. The reach closure (R² 0.92 Blue, 0.82 Red,
   after adding formation width) is the bottleneck: a Level A limit (RQ2).
6. **Stability:** the withdraw term flips between `Rb*wd` and `Ab*wd` across bootstrap refits
   (53% / 47%). With control x alive candidates excluded it is kept 12 of 12 times.
7. **Counterfactual ranking (notebook 3; cf 526 states, cftrain 1,803):**
   - Best single intent wins about 51%, best of six in hindsight 69%; the intent decides the winner
     in 47% of states. withdraw/nearest is the best choice in 43% of states, advance in about 53%.
   - Not chaos at the level of reruns: withdraw/focus and withdraw/nearest (same behaviour) give
     identical outcomes in 100% of states. advance/focus vs advance/nearest differs by std 0.09.
   - The SINDYc models rank no better than "always advance" (win 51%, rank correlation 0.16-0.18).
   - **Learned directly from cftrain** (gradient boosting on the start state, predict the
     difference between intents, pick the best of three): **58.6% win on cf** against 51% for
     always advance, regret 0.065 against 0.094. withdraw - advance: hold-out R² 0.10 (team) /
     0.13 (team + unit features); hold - advance 0.64; focus - nearest about 0.05.
   - Unit-level features add almost nothing (choice win rate 58.2% vs 58.6%).
   - Correction: the earlier "R² below 0" for withdraw - advance came from 499 states only.
8. **Planner in the real game (400 close battles, same seeds):**

   | policy | Blue wins |
   |---|---|
   | hold/nearest | 11.5% |
   | random switching | 46.0% |
   | planner, rich model | 57.2% |
   | advance/nearest | 59.8% |
   | advance/focus | 67.0% |
   | **planner, sparse model** | **69.8%** |
   | oracle planner (real simulator) | 94.2% |

   Standard error about 2.5 points. The sparse planner beats advance/nearest clearly (wins 63
   battles it loses, loses 23 it wins) but is **not clearly better than always advance/focus**.
   The oracle shows the scheme could reach 94%: the gap is the model.
   - **Paired on the same seeds**, sparse planner - advance/focus = +2.8 points, bootstrap 95%
     interval [+0.5, +5.2] (wins 17 battles advance/focus loses, loses 6 it wins).
   - **What it chose:** 84% advance/focus, 11% hold/focus, 5% withdraw (on average over battles).
     It withdraws at some point in 30% of battles; there it wins 75% vs 66% for advance/focus.
     The oracle uses advance/nearest, advance/focus and withdraw about a third each.

## Read with care

- The sparse planner's edge is mostly "use focus fire" plus a small, statistically detectable gain
  from occasional withdraw/hold.
- The rich model plans *worse* than the sparse one although it predicts slightly better.
- Thresholds, closure form and planner settings were chosen on validation data and one planner run.
- The learned choice in notebook 3 is evaluated on cf states, not by playing it in the game.

## Not done

- No tests for the new modules (the existing 47 pass). Nothing committed.
- The learned choice rule is not played in the game (would need a policy wrapper and an
  `eval_planners.py` run).
- Oracle with perturbed copies (is the 94% planning or foresight?) not run.

## Added 11 Nov: SINDy-chosen switches and transfer to stalkers

`notebooks/sindyc_04_switches_and_transfer.ipynb`, `models/switch_library.py`, exp07 (stalkers,
1,500 + 300 battles, `scripts/collect_stalkers.py`; `load_team_states` now takes `reach` and
`unit_health`).

- **Switches as library candidates.** Plain STLSQ over a grid of reach switches fails (37 terms,
  opposite-sign pairs, val R² 0.44). With physical signs: 4 terms at 5.9-6.9, R² 0.76. Best subset
  (one switch per mechanism): fire switch 6.4 for both sides (1.11 x reach).
- **Accuracy:** health at 4 s 0.058 vs 0.057 with hand closures; distance worse (0.56 vs 0.49).
  Parameters fitted outside SINDy: 10 grid choices instead of about 40 curve_fit parameters.
- **Dimensionless coefficients, marines vs stalkers (switch model):** firepower x tau 0.56 / 0.53
  and 0.67 / 0.63; fire switch / reach 1.11 / 0.94-0.97; hold and withdraw closing 1.0-1.1 / 0.96
  walking speeds; hold / withdraw switch 1.4-1.5 reach in both. **Differs:** advance closing 1.9 vs
  0.9 walking speeds; alive exponent 0.5 vs 0.37-0.40 (shields); the withdraw fire term is not
  selected for stalkers.
- **Transfer, no refit (stalker validation, health at 4 s):** marine model as is 0.090, scaled by
  reach / tau / speed **0.030**, refit on stalkers 0.025, persistence 0.074. Scaling closes 93% of
  the gap (82% with hand closures).

## Added 12 Nov: hold the line, and everything re-collected

`hold` used to mean "units with nothing in range stand still", so half the team never fought
(fixed hold/nearest won 11.5%). It now means **hold the line**: don't approach; once any Blue unit
has an enemy in range, every unit closes in and fights as under advance (`envs/intents.py`).
exp06 and exp07 were re-collected with it (same seeds and sizes); the first collection is kept in
`data/*/exp06_intents_oldhold` and `exp07_stalkers_oldhold`. The sparse model now leaves out the
control x alive candidates (`exclude=ALIVE_CONTROL_TERMS`). All notebooks re-run.

- **Hold the line is the best fixed intent**: planner test (400 battles) hold/focus 70.2%,
  advance/focus 67.0%, hold/nearest 61.5%, advance/nearest 59.8%. Pilot: hold/nearest 59.7%
  against 55.3%; from 12-17 apart hold 55% against advance 35%.
- **The model is unchanged**: `(health_red)' = -0.186 Rb + 0.163 Rb*wd`, `close*hold` -3.29;
  health error at 4 s 0.057 (persistence 0.23, no control 0.080, generic SINDYc 0.053 with 138
  terms, real reach 0.028).
- **Correction:** the withdraw term does not need 6,000 battles. With the restricted library it
  is found with 30 battles (coefficient 0.141, 0.161 from 100 on). The earlier result came from
  `Rb*wd` and `Ab*wd` competing.
- **The planner's gain is gone:** sparse planner 67.2%, the same as always advance/focus (+0.3,
  interval -0.7 to +1.3) and below always hold/focus (-3.0, interval -6.5 to +0.5). It plays
  advance/focus 61%, hold/focus 38%. Oracle 94.8%.
- **Why the model can't use hold:** hold and advance differ only before contact, and the model
  sees that difference only as slower closing (-3.3 against -6.0). That holding wins more must
  come from something the team state doesn't carry (how Red arrives: stretched out, a few at a
  time). In the counterfactual sets (branch mostly in contact) hold and advance give the same
  outcome in 91% of states.
- **Counterfactual ranking:** best fixed 58%, hindsight 74%, SINDYc picks 58%, learned from
  cftrain 63.6% (team) / 64.9% (team + units).
- **Transfer to stalkers holds:** health at 4 s as is 0.095, scaled 0.029, refit 0.026 (95% of
  the gap closed). The hold switch doesn't transfer (stalker `close*hold` nearly 0).
