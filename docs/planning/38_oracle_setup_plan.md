# Is the setup the limit? The oracle with the planner's settings, and an exploitation check (plan, 22 Nov 2026)

**Status: approved by Knut 22 Nov 2026, as drafted; rules fixed before any battle.**

## Why

`37` concluded that better prediction accuracy did not improve planning: a black-box model that predicts far better
planned worse than the SINDy models. Two explanations remain open:

| explanation | test |
|---|---|
| **the setup** (near-ties, a 3 s horizon, a dominant per-unit rule) leaves little to gain whatever the model | a **perfect model** with the planner's settings: does it beat fixed kite? |
| **model exploitation:** the planner picks intents whose predicted gain is a model error, more so with a black box | does each planner's chosen intent do worse than predicted, and by how much? |

## Part 1: the oracle with the planner's settings

The exact oracle (`OraclePlannerPolicy`: plays each candidate in a copy of the real game, no noise) with:

| | oracle7-h3 | oracle7-h8 |
|---|---|---|
| candidates | the five + kite/nearest, kite/focus | same |
| re-plan / commit | 1 s / 2 s (as the world-model planners) | same |
| horizon | **3 s (as the planners)** | 8 s (as Phase A) |
| score | health margin at the horizon (as the planners) | same |

- **Scenarios:** S2, S4, S5 (the main scenarios of `34` / `36`).
- **Seeds:** `34`'s seeds, 21 x00 000 +, 400 per scenario, paired with every policy already played there.
- **Run time:** timed on 5 battles per oracle first. Estimate 3.5-4 h on 8 workers, so it runs overnight with the
  lid open.

## Part 2: exploitation check

The three world-model planners (frozen `model:h3`, `musigma:h3`, `mlp:h3`) replayed on the first 100 seeds of S2, S4
and S5 (the same battles as before: the planners are deterministic). At every re-plan, two numbers are logged for the
chosen intent:
- **predicted:** the gain in health margin over the next 1 s, from the model's rollout;
- **realised:** the gain over the 1 s actually played, until the next re-plan.

**Optimism** = mean(predicted − realised), per planner and scenario, with a battle bootstrap. The same is also
computed for the average over all candidates, so that the optimism of the *chosen* intent can be separated from a
plain bias of the model. A deterministic replay check (the logged wins equal `34` / `37`'s) runs first.

## Rules (fixed once approved)

| | comparison | read as |
|---|---|---|
| **O1 (main)** | oracle7-h3 − best fixed kite (as `34`), S2, S4, S5 | beats / ties / loses, paired 95% bootstrap |
| O2 | oracle7-h8 − best fixed kite | the same |
| O3 | oracle7-h3 − μ, σ planner | the same (how much a perfect model would add) |
| E | optimism of the chosen intent, per planner | reported with 95% intervals; no pass/fail |

**Interpretation, fixed now:**
- **O1 never beats kite** (ties or loses in all three): the setup claim holds. Even a perfect model with the
  planner's settings does not beat the dominant rule. `37`'s wording stands.
- **O1 beats kite in at least one scenario and loses in none:** the claim is wrong for these settings. A perfect
  model would help, so model accuracy matters, and the black box failed for another reason. `37`'s wording and the
  supervisor doc are corrected.
- **O1 mixed:** reported per scenario, and the wording is narrowed to match.
- **O2 beats where O1 does not:** the horizon is the limiting part of the setup.
- **E:** if the MLP's chosen-intent optimism is clearly larger than the SINDy planners' (intervals not
  overlapping), exploitation is reported as a measured factor, not a guess.

## Code needed

- Oracle names "oracle7-h3" and "oracle7-h8" in `headroom.make_policy` (`OraclePlannerPolicy` with the seven
  candidates and the given horizon).
- Run through `phase_b.run` with prefix `closed34_`, so rows pair with the existing ones.
- In `PipelinePlannerPolicy`, log at each re-plan the chosen intent's predicted 1 s gain, every candidate's 1 s
  gain, and the margin at the re-plan (realised gains follow from consecutive re-plans). It is off by default and
  writes to a separate file.

## Expectations (written before any battle)

- **O1:** ties or loses in S2 (kite is at 89.5%, and the five-candidate 8 s oracle reached 96.5% on Phase A seeds); in
  S4 about 50/50 (the 8 s oracle with kite reached 79% vs kite 34% on Phase A seeds, but 3 s is much shorter); S5
  ties. I expect the setup claim to hold only partly.
- **O2:** beats kite in S4 and S5.
- **E:** all planners are optimistic about their chosen intent; the MLP the most.

## Open questions for Knut

| | question | draft answer |
|---|---|---|
| Q1 | All 400 seeds, or 200 to halve the time? | 400: pairs fully with `34` / `37`; overnight anyway |
| Q2 | Include S1 and S3 (flagged)? | no: kite is at 97-99% there, so there is no room to show anything |
