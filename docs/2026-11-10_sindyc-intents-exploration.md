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
