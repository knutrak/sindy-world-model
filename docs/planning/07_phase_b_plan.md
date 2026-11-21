# Phase B: does the world model capture the headroom? Pre-registration (draft, 21 Nov 2026)

**Status: approved 21 Nov ("run the whole phase B"): B1-B5 as recommended. B6, Knut's own expectation, was
not written before the run. Changes are dated amendments at the end, written before the results are read.**

## Question

Phase A showed that choosing intents well is worth a lot in all five scenarios: the exact oracle beats the best
fixed intent by 23-55 win points, the perturbed oracle by 15-38 (`01_headroom.md`). Phase B asks the RQ5
question with the actual world model:

> A planner that chooses intents with the frozen pipeline model: how much of that headroom does it capture?

This is the evidence RQ5 needs before an LLM agent is put on top of the world model.

## What we already know (before any Phase B run)

From the counterfactual sets (`02`-`06`), single decisions in S1 and S2:

- the frozen model ranks the five intents **worse than always advance/nearest at 1-2 s** (regret 0.017 vs 0.005
  in S1) and ties the best fixed intent at 8 s;
- it **picks a focus intent in about 98% of states**: its only focus term (Red's alive law) has no short-term
  cost, while in the game focus costs about 11% of Blue's firing at first;
- judged by the battle's outcome, its picks win **as often as the best fixed intent** (S1 58-59% vs 58.4%; S2
  61-62% vs 61.2%); a perfect model with a 4 s horizon would win 67% / 66%;
- adding a focus cost, focus terms in d' or shape dynamics does **not** improve this (`04`, `06`).

The earlier closed loop (10 Nov, the older team model, marines) won the same as always advance/focus while
playing advance/focus 61% and hold/focus 38%.

## The planner

`PipelinePlannerPolicy` (new, in `envs/planner.py`), the same scheme as `ModelPlannerPolicy` and the oracle:

- every **1 s** it reads the per-type state from the live game and rolls the model forward once per candidate;
- each candidate (the five of Phase A: advance/nearest, advance/focus, hold/nearest, hold/focus, withdraw) is
  played for the **commit** time, then advance/nearest to the end of the **horizon**;
- it plays the candidate with the best predicted **health margin** (Blue − Red) at the horizon, the oracle's
  score;
- no switch margin: the best candidate is played even if it differs from the current one by a little.

Models, frozen (`docs/generalisation/06_frozen_equations.md`, `generalisation_frozen.pkl`), no refit:

| scenario | model | note |
|---|---|---|
| S1 marines | marines | as trained |
| S2 mixed, same mix | mixed | as trained |
| S3 Blue marauders vs Red marines | mixed | composition outside what the mixed model was tested on |
| S4 underdog (mixed, 0.6-0.8 strength) | mixed | strength ratio outside training |
| S5 uneven mixes | mixed | as the held-out `uneven` test (fitted on mixed) |

### Settings (fixed now)

| | primary | secondary (reported, not used for the main claim) |
|---|---|---|
| horizon | **3 s** (inside the reliable 2-3 s) | 2 s and 4 s |
| commit | **2 s** (as the oracle) | — |
| re-plan | **1 s** | — |
| model | **frozen** | focus-cost variant (`04_focus_variant.md`, "compensated"), exploratory |

With commit 2 s and horizon 2 s the candidate is played for the whole rollout.

### Implementation steps (before any Phase B seed is played)

1. A live per-type state: `typed_state_from_game(game, types)` giving exactly the columns of
   `states.typed.load_typed_states` (health, alive, reach per side and type, fronts, depth, width, distance; reach
   capped at 6).
2. **Consistency test:** on 20 recorded `val` battles, the live state at every step equals the recorded table
   (to float precision). If it doesn't, Phase B waits.
3. `PipelinePlannerPolicy` with unit tests: on a recorded state its five predictions equal `Pipeline.rollout` on
   the same batch.
4. A 20-battle smoke run per scenario on seeds outside Phase B's range (19 700 000 +), for timing and crashes only;
   its results are not read beyond that.

## Seeds and runs

New seeds, used nowhere else: **S1 14 000 000 +, S2 14 100 000 +, S3 14 200 000 +, S4 14 300 000 +,
S5 14 400 000 +** (Phase A used 13 x00 000 +).

| policy | seeds per scenario | why |
|---|---|---|
| model planner, primary (frozen, horizon 3 s) | 400 | the main result |
| model planner, horizon 2 s and 4 s | 400 each | horizon sensitivity |
| model planner, focus-cost variant (horizon 3 s) | 400 | exploratory |
| the six fixed intents, random switching | 400 | the baselines on the same seeds |
| exact oracle | first 200 | the headroom on these seeds |

Cost: model planners and fixed intents are ordinary-speed battles (minutes in total). The exact oracle is about
26 s per battle: 5 × 200 battles, roughly 1 h on the laptop's 9 workers. No perturbed oracle in Phase B: the
shared-noise rerun of it is a separate Phase A correction, on Phase A's seeds.

## Measures and read-out (fixed now)

**Primary outcome:** O1, battles won. **Primary comparison:** model planner (primary settings) − **the best fixed
intent from Phase A**, chosen before Phase B, so the baseline isn't picked on Phase B's seeds:

| | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| best fixed intent (Phase A, O1) | hold/focus | advance/focus | hold/focus | hold/focus | advance/focus |

Paired over the 400 seeds, bootstrap 95% interval. Read-out:

- **beats** the best fixed intent: interval entirely above 0;
- **ties**: interval contains 0;
- **loses**: interval entirely below 0.

**Share of headroom captured**, on the oracle's 200 paired seeds:

```
share = (model planner − best fixed) / (exact oracle − best fixed)      (O1; bootstrap 95% interval)
```

**Main scenarios: S1, S2, S5** (the model is inside its tested range). S3 and S4 are reported alongside, flagged:
a failure there may come from the model being outside its range rather than from planning.

**Secondary (reported, no claims of their own):** O2 health margin, O4 exchange ratio (log), O3 units preserved
(only next to O1); the best fixed intent *on Phase B's seeds* as a check on the baseline; the horizon 2 / 4 s
planners; the focus-cost variant.

**Diagnostics, logged per re-plan:** the five predicted scores and the choice. Reported: intent use (share of
re-plans per intent), how often the choice switches, and the predicted gap between the best and second-best
candidate.

## Expectations (written before the run)

From the counterfactual evidence above:

| | expectation |
|---|---|
| primary comparison, S1 / S2 / S5 | **ties** the best fixed intent (within about ±3 points) |
| share of headroom captured | about 0 (interval roughly within −0.15 to +0.15) |
| intent use | focus intents in more than 90% of re-plans; withdraw rare |
| horizon 2 / 4 s | no better than 3 s |
| focus-cost variant | plays mostly advance/nearest, so it ties or loses slightly against the best fixed intent (advance/nearest is below it in S1, S3, S4) |
| S3, S4 | no reliable expectation; read with the range caveat |

**Knut's expectation:** *(to fill in before approval: e.g. "the planner beats the best fixed intent in S1 because
it re-plans every second", with the size you expect)*

A result that would **surprise** this plan: the primary planner beating the best fixed intent in any of S1, S2, S5
with the interval above 0. If that happens, the diagnostics should show which decisions differ from the fixed
intent and when.

## What is not allowed

- Tuning anything (horizon, commit, score, candidates, model) on Phase B's seeds.
- Re-running on Phase B's seeds with changed settings and reporting that as the primary result.
- Using the held-out sets (`test`, `test2`, `test3`, `test4`, `exp10` held-out; all spent).

## Outputs

- `docs/planning/08_phase_b.md` with the tables, the diagnostics and the deviations, if any;
- scripts in `experiments/planning/` (`phase_b.py`, `analyse_phase_b.py`);
- data in `data/processed/planning/phase_b_<scenario>.csv`.

## Decisions for Knut

| | question | my recommendation |
|---|---|---|
| B1 | primary horizon 3 s with commit 2 s, or another pair? | 3 s / 2 s; 2 s and 4 s as secondary |
| B2 | baseline: the Phase A best fixed intent (fixed now), or the best on Phase B's seeds? | Phase A's, fixed now; the other as a check |
| B3 | main scenarios S1, S2, S5, with S3 and S4 flagged? | yes |
| B4 | include the focus-cost variant as an exploratory arm? | yes; it costs minutes |
| B5 | 400 seeds for the planners, 200 for the exact oracle? | yes |
| B6 | your own expectation (above) | write it before approving |

## Amendments (before any Phase B seed is played)

- **21 Nov, implementation step 2.** The consistency test compares the planner's single-step live state with
  the per-type table built from the same battle's full snapshot record (`typed_states` over all steps), on
  battles played live in the test, instead of re-playing recorded `val` battles. The recorded sets were built
  with the same snapshot logger and `typed_states`, so this checks the one new path (one live snapshot → state
  → batch) without having to re-create the scripts' intent schedules.
