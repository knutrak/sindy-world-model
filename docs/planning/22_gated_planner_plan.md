# A confidence-gated planner: pre-registration (21 Nov 2026, overnight run)

**Status: written before the threshold is computed and before any battle of this test is played. Knut started the
overnight run, which includes this test, on 21 Nov.**

## Why

Most of the planner's decisions are near-ties: the predicted gap between its best and second-best candidate has a
median of 0.000-0.001 (team margin, ten units of the first type), below the model's own error (`20`). In S1 it acts
on those differences and ends 2-4 points below hold/focus over three sets of battles (`21`, section 2). Its valuable
choices come with larger predicted gaps: in S3, withdrawals have a median gap of 0.010.

## The planner

`GatedPlannerPolicy`: the Phase B planner (frozen model, horizon 3 s, commit 2 s, re-plan 1 s, five candidates) with a
**default intent** D and a **threshold** τ. At each re-plan, with predicted scores s_i:

```
play argmax_i s_i   if  max_i s_i − s_D > τ
play D              otherwise
```

- **D = Phase A's best fixed intent** in the scenario (S1 hold/focus, S2 advance/focus, S3 hold/focus, S4 hold/focus,
  S5 advance/focus), chosen before Phase B.
- **τ = the model's typical error in the difference between two candidates' scores:** the median, over the
  counterfactual states and all pairs of distinct candidates, of |(predicted s_i − s_j) − (true s_i − s_j)| at 3 s,
  where s is the change in the team health margin (as `02`). One τ per model: from exp06 cf for the marines model
  (S1), from exp08 cf for the mixed model (S2-S5). Twin pairs (hold and advance with the same target) count as
  distinct candidates here; nothing else is chosen from these data.

The frozen model is unchanged; only the decision rule differs.

## Test

Fresh seeds **18 x00 000 +** (S1-S5), 400 each. Arms on the same seeds: the gated planner, the Phase B planner
(ungated), the six fixed intents and random switching.

| read-out (wins, paired, bootstrap 95%) | main scenarios | flagged |
|---|---|---|
| gated − D (the best fixed intent) | S1, S2, S5 | S3, S4 |
| gated − ungated planner | S1, S2, S5 | S3, S4 |

beats / ties / loses on the interval, per scenario, no pooled claim. Also reported: how often the gate lets the
planner deviate from D, and which intents it plays then.

## Expectations (written before)

- S1: gated ties or slightly beats hold/focus, and beats the ungated planner.
- S3: keeps most of the ungated planner's gain over hold/focus (+10 points or more).
- S2, S4, S5: ties.

## Not allowed

Changing τ, D or the planner after seeing these battles; reruns with another τ are reported as such.

## Amendment (after τ was computed, before any battle of this test)

τ = 0.033 (marines model) and 0.042 (mixed model); quartiles 0.005 / 0.075 and 0.006 / 0.091
(`data/processed/planning/gated_tau.json`). Noted before the run: the ungated planner's S3 withdrawals had a median
predicted gap of 0.010 (`20`), below τ, so the gate is likely to suppress most of them and the S3 expectation above
may fail. The rule stays as registered.
