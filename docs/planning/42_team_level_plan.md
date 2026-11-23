# Planning at team level only: the μ, σ planner and a matched oracle without kite (plan, 23 Nov 2026)

**Status: approved by Knut 23 Nov 2026 ("do A"); rules fixed before any battle.**

## Why

Kite is a micro-intent: it decides what each unit does between shots. Advance, hold and withdraw are team-level
decisions. With kite among the candidates, it swamped every comparison (`41`). This test plans at team level only,
the setup of Phase A and B: five candidates, advance / hold / withdraw × nearest / focus, against the scripted Red.

Two pieces are missing for a complete team-level RQ5 table:
- the μ, σ planner restricted to the five team-level candidates;
- a perfect model with the planners' settings (the exact oracle, five candidates, re-plan 1 s, commit 2 s, horizon
  3 s and 8 s) on the same seeds.

## Policies (new)

| name | what |
|---|---|
| `musigma5:h3` | the μ, σ planner of `34` (same model, clipped), with the five team-level candidates only |
| `oracle5-h3` | exact oracle, five candidates, re-plan 1 s, commit 2 s, horizon 3 s |
| `oracle5-h8` | the same with horizon 8 s (Phase A's oracle) |

**Existing on the same seeds (`34`):** the frozen planner (`model:h3`, five candidates) and Phase A's best fixed
team-level intent (S1, S3, S4 hold/focus; S2, S5 advance/focus).

**Seeds:** `34`'s, 21 x00 000 +, 400 per scenario, all five scenarios, paired.

## Rules (fixed now)

| | comparison | read as |
|---|---|---|
| **T1** | μ, σ planner (5) − best fixed team-level intent | beats / ties / loses, paired 95% bootstrap, per scenario |
| T2 | μ, σ planner (5) − frozen planner | the same |
| T3 | 3 s oracle (5) − best fixed team-level intent | the same: the matched ceiling |
| T4 | 8 s oracle (5) − best fixed team-level intent | the same |
| share | (μ, σ planner − best fixed) / (3 s oracle − best fixed) | the share of the matched ceiling captured, per scenario |

**Statement:** "team-level planning with the μ, σ world model beats the best fixed team-level intent" holds if T1
beats in at least one of S1, S2, S5 (Phase B's main scenarios, inside the training range) and loses in none of them.
S3 and S4 are reported, flagged as in Phase B.

## Run time (estimated)

| step | time |
|---|---|
| μ, σ planner, 2,000 battles | about 3 min |
| oracles, 4,000 battles | 30-60 min (timed on a smoke run first) |

## Expectations (written before any battle)

- **T1:** the μ, σ planner beats the best fixed team-level intent in S2 and S3, ties S5, loses or ties S1.
- **T3:** the 3 s oracle beats the best fixed team-level intent everywhere.
- **Share of the matched ceiling captured:** 20-50%.
