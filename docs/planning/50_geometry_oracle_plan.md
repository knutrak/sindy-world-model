# Perfect geometry, SINDy damage: how much of the ceiling would the world model capture? (plan, 23 Nov 2026)

**Status: run on Knut's instruction of 23 Nov ("continue and do additional research if you find it intriguing and
feasible"); the read-out below was fixed before any battle.**

## Why

Every planning result points to geometry (who is in range of whom) as the world model's weak point. The
unit-level models now reach the closure's accuracy but not beyond (`generalisation/21`). Before more work goes
into geometry, one question decides whether it is worth it:

> If the geometry were perfect, would the SINDy world model plan close to the ceiling, or are its damage and death
> equations a second bottleneck?

## The geometry oracle (`geo5-h3`)

The same planner settings as `oracle5-h3` (five team-level candidates, re-plan 1 s, commit 2 s, horizon 3 s). For
each candidate at each re-plan:
1. play the candidate in a copy of the real game, as the oracle does, and record at every step how many units of each
   side and type have an enemy within reach (the typed states' rule);
2. roll the **frozen SINDy pipeline** forward from the live state with the candidate's controls, with its reach
   replaced by that recorded reach. Everything else comes from the model: health, units alive, distance;
3. score by the model's predicted health margin at 3 s, as the world-model planners do.

So the geometry is perfect, and damage and deaths are the SINDy model's. One caveat: the recorded reach also reflects
the true deaths in the copy, so it carries some true information about deaths. That makes this an upper bound for
"perfect geometry".

## Runs

S1-S5, `34`'s first 200 seeds per scenario, paired with the existing `oracle5-h3`, best fixed team intent, frozen
planner and μ, σ planner rows. About 1,000 battles at the oracle's cost: about 5-10 min.

## Read-out (fixed now)

**Share of the matched ceiling captured** = (geo5-h3 − best fixed) / (oracle5-h3 − best fixed), per scenario, with
paired 95% intervals on both differences.

| statement | holds if |
|---|---|
| **"geometry is the bottleneck"** | share ≥ 0.7 in at least three of S1-S5 |
| "damage and death equations are a second bottleneck" | share ≤ 0.3 in at least three |
| mixed | otherwise; reported per scenario |

## Expectations (written before any battle)

Share about 0.5-0.8 in S1-S3; lower in S4 (outside the training range) and S5. About 45% that "geometry is the
bottleneck" holds.
