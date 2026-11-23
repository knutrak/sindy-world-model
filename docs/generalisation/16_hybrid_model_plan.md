# A unit-level hybrid world model in the planner (plan, 23 Nov 2026)

**Status: approved by Knut 23 Nov 2026 ("continue the work and see if it helps the planner"); criteria fixed before
any rollout.** Exploratory groundwork for next semester's RQ5.

## Why

`15`: per-unit movement follows a hybrid law. Each unit walks at 3.15, heading by intent, or stops, and the switch
is predicted at 95-97% for marines by a sparse rule. Mixed armies fail on crowding. A world model built on this
law tracks unit positions, so reach is **counted**, not guessed. That is the geometry the planners got wrong
(`planning/45`).

## The model (marines main; mixed flagged)

Every unit carries position, health and alive. One step (0.125 s):

1. **Target.**
   - Blue follows its intent's target rule: nearest enemy, or focus (the team picks the weakest enemy any Blue unit
     can reach).
   - Red picks its nearest enemy.
2. **Mode.** Walk or stop, from the switch classifier of `15`: L1 logistic, refitted per side and type on train.
   Mixed Blue marines get the depth-3 tree, as no ≤ 8-term logistic rule exists.
3. **Walk.** Speed 3.15 (`15`). Heading by intent, from `15`'s measured headings:
   - towards the target for advance, hold and Red;
   - away from the nearest enemy for withdraw;
   - for kite, away while reloading, else towards.
4. **Damage.** A unit whose target is within its reach deals k × (the target type's hp) per second. k is the frozen
   pipeline's firepower for that attacker and target type; nothing is refitted.
5. **Death** at 0 health.

**Not modelled:** collision avoidance (units may overlap), and Red's target stickiness. Both are stated
simplifications.

**Data:** fit on train only (the `14` rows). Evaluate on val and the counterfactual sets. No held-out data, no
collection.

## Stages and criteria (fixed now)

| stage | what | passes if |
|---|---|---|
| **H1 geometry** | as `13` U2: move units by the hybrid law, removing them at their recorded death times; reach counted | team reach error ≤ 0.7 × the frozen closure's at 2 s and 4 s (exp06 val; paired intervals by battle) |
| **H2 full model** | the whole model from recorded states with Blue's recorded intents | margin error below the frozen model's at 2 s and 4 s (paired, below 0) |
| **H3 ranking** | the five team-level candidates on exp06's counterfactual sets (as `11` M4) | regret below the frozen model's at 2 s and 4 s (paired, below 0); also reported against μ, σ and always advance/nearest |
| **H4 closed loop (S1)** | the hybrid planner (five candidates, re-plan 1 s, commit 2 s, horizon 3 s) on `34`'s first 200 S1 seeds | paired with the existing rows: against hold/focus (best fixed), the μ, σ planner, and `oracle5-h3`. Read-out: share of the matched ceiling captured; **"helps the planner"** if it beats the μ, σ planner (paired interval above 0) |

The stages run in order. Each later stage runs only if the earlier one passes; a failure is reported with its
reason. Mixed (S2) is run alongside H1-H3 as a flagged extra, with the same read-outs and no ruling.

## Run time (estimated)

| step | time |
|---|---|
| code | a few hours |
| H1-H3 | minutes |
| H4 | about 15-30 min (unit-level rollouts per re-plan; timed first) |

## Expectations (written before any rollout)

- **H1:** passes for marines (reach error 40-60% below the closure).
- **H2:** about 60%. Overlapping units and simplified targeting will blur the damage.
- **H3:** about 50%.
- **H4:** if reached, captures 20-50% of the S1 ceiling, against the μ, σ planner's −4%.
