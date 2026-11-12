# Status and next steps (11 Nov 2026)

Replaces `2026-10-30_where-i-left-off.md` as the place to start. Details:
`notebooks/00_findings_summary.ipynb` (ranked findings), `2026-11-10_sindyc-intents-exploration.md`
(the SINDYc work), `docs/findings.md` (F1-F26, not yet updated with anything below).

## Where I stand, in short

- **A world model with control exists.** Sparse SINDYc on team state (health, alive, distance) with
  Blue's intent as one-hot control (advance / hold / withdraw x nearest / focus). About ten
  readable terms, e.g. `(health_red)' = -0.185 Rb + 0.164 Rb*wd` (withdrawing removes 89% of
  Blue's fire). Health error at 4 s 0.056 against 0.22 for persistence and 0.079 without control.
- **SINDy now chooses the switches** (reach and movement), instead of hand-fitted closures: same
  health accuracy, worse distance (`sindyc_04`).
- **It transfers.** In unit-free form (reach, time to kill, walking speed) the marine model predicts
  stalker battles without refitting: health error at 4 s 0.030, against 0.025 refitted and 0.090
  unscaled. Firepower and switch distances match across unit types; advance closing speed and the
  alive exponent don't.
- **Limits.** The accuracy limit is the team-level reach closure (real reach halves the error). The
  model predicts well on average but can't rank intents state by state.
- **Update 12 Nov: hold the line.** `hold` was fixed (it left half the team out of the fight) and
  all exp06 / exp07 data re-collected. Holding the line is now the best fixed intent (hold/focus
  70.2% against advance/focus 67.0%), and the model planner (67.2%) no longer beats always
  advance/focus nor always hold/focus: the team-level model can't see why waiting wins. Details
  at the end of `2026-11-10_sindyc-intents-exploration.md`.
- **All of exp06 / exp07 is exploratory:** collected with uncommitted code (`allow_dirty`), models
  chosen while looking at validation numbers, no intervals.

## Phase 1 status (milestone end of 2026)

| Step (project description, section 5) | Status |
|---|---|
| 1 Understand the environment | Done (F1-F3) |
| 2 Collect and analyse data | Done: exp01-exp05; exp06 (intents), exp07 (stalkers) exploratory |
| 3 First world model (Level A, SINDy) | Done |
| 4 Refine: states, libraries, methods | Largely done: reach, distance, control, switch library; no discrete-time model, no Level B |
| 5 Validate on unseen data vs. baselines over horizons | Done without intervals |
| 6 Generalise across scenarios | Started: one second unit type (stalkers), force size within exp05/exp06 |
| 7 Clean world-model interface | Partly: `IntentModel.simulate`, `state_from_game`, a planner that uses them |

| RQ | Status |
|---|---|
| RQ1 interpretable world model | **Mostly answered**: control terms with physical readings; caveat: the switch geometry is designed, SINDy chooses within it |
| RQ2 state representation | **Partly**: Level A variants compared; the reach closure is the measured limit; unit-level features didn't help choosing; no Level B model |
| RQ3 accuracy over horizons | **Answered without intervals** (1-8 s, against persistence, last slope, no control, generic SINDYc) |
| RQ4 generalisation | **Started, promising**: marines → stalkers by scaling; one unit type, five properties changed at once |
| RQ5 does it help an LLM agent | Not started (Phase 2). Decided: the LLM chooses among named intents and gets the model's predictions per intent |

## What to do next, in order

1. **Consolidate (this week).** Re-collect exp06 and exp07 with committed code; tests for
   `intents`, `planner`, `intent_model`, `switch_library`, `geometry`; add F27+ to
   `docs/findings.md`. Decide the core model (proposal: sparse SINDYc with SINDy-chosen switches, in
   unit-free form).
2. **One unit property at a time (RQ4, RQ1).** Custom marines with only longer reach, only higher
   speed, only more health (SMAClite custom unit JSON), a few hundred battles each. Does the scaled
   model predict each one? Also measure the advance closing speed directly in the data (marines 1.9
   vs stalkers 0.9 walking speeds).
3. **Intervals.** Bootstrap by battle on the horizon errors and the transfer table.
4. **Write the preparation thesis** around RQ1-RQ4 once 1-3 are in.

`hold` is fixed (hold the line, 12 Nov). New open question: why does waiting win, and what would
the state need to contain for the model to see it (RQ2)? Leave phases, MPPI and new intents for now.

## For the supervisors

- MPPI: with a few discrete intents, exhaustive search is exact; MPPI only pays off for long intent
  sequences or with a model ensemble. What do they want it for?
- One-hot intents as the control input, chosen because the LLM picks named intents. Agree?
- Is RQ4 as "transfer by unit properties" the right framing?
