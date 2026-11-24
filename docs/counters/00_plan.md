# Counters from game mechanics: a testbed where no strategy dominates (plan, 24 Nov 2026)

**Status: approved by Knut 24 Nov 2026 ("branch out and experiment with A … including recollecting data and new
intents if needed").** Work on git branch `counters`. Stages run in order; each stage's gate decides whether the
next runs. The rules are fixed here before any battle.

## Why

In marines and marauders against a scripted (or mirrored) Red, one strategy, kite, is best against every opponent
behaviour (`planning/41`). So a planner has nothing to adapt to. Real tactics have counters that come from unit
mechanics:
- faster units cannot be kited;
- splash damage punishes clumping;
- longer range punishes closing in slowly.

The aim is a testbed where **the best Blue strategy depends on what Red does, and vice versa**. Then, a world-model
planner that reads the situation can beat any fixed rule.

## Stage 1: payoff screens (simulator only, no data collection)

**Candidate match-ups** (Blue against Red, on `mixed_m_mrd`, open terrain):

| id | Blue | Red | counter mechanics |
|---|---|---|---|
| Z | 7 marines, 3 marauders | 12 zerglings, 4 banelings | zerglings outrun kiting marines; banelings' splash (radius 2.2) punishes clumps |
| P | 7 marines, 3 marauders | 4 stalkers, 3 zealots | stalkers outrange marines and outrun them; zealots soak damage in melee |

Counts are calibrated first, so that both sides advancing with nearest targets gives Blue a 35-65% win rate
(100 battles per setting).

**Strategy menu (both sides, through the mirrored Red of `planning/40`):**
- advance, stand (hold position), withdraw, kite, each with the nearest target rule;
- advance with focus;
- **spread**, a new intent: advance, but a unit with a friend within 2.2 (the baneling splash radius) that cannot
  fire this step moves away from that friend.

That is 6 Blue rows × 6 Red columns, 150 paired seeds per cell, fresh seeds 24 x00 000 +.

**Read-outs (as `planning/41`):**
- **P1:** no Blue row within 5 points of the best Blue row against every Red column;
- **P2:** the adaptation headroom (mean over Red columns of the column-best Blue win rate, minus the best single
  Blue row's mean), in points;
- the best-reply structure: Blue's best reply per Red column, Red's per Blue row, and whether a cycle exists.

**Gate 1:** a match-up qualifies if **P1 holds and P2 ≥ 10 points**. If none qualifies, the work stops at stage 1 and
reports what dominates and why.

## Stage 2: data (only for a qualifying match-up)

As exp08:
- both sides switch intents independently every 1-4 s, from the menu above; Red gets its own random stream;
- train 4,000, val 500 and counterfactual sets (each Blue intent for 4 s from a drawn moment, 300 seeds);
- per-type state tables.

About 30-60 min of compute.

## Stage 3: the world model

Extend the pipeline to more than two unit types per side, and add Red's intent as a control. That includes:
- melee and splash targeters in the firepower terms;
- a test that the two-type case is unchanged.

Fit on train; check on val against persistence and the frozen two-type model's structure. **Gate 3:** rollout
errors at 2 s no worse than the pipeline's on exp08 (relative to persistence).

## Stage 4: planning

Matched-settings oracle, world-model planner and best fixed strategy against each Red strategy and against a
switching Red. Planned in detail after gate 3, with its own rules.

## Expectations (written before any battle)

- **Z:** spread beats advance against banelings; kite loses to zerglings. A cycle is plausible. About 50% that Z
  qualifies.
- **P:** stalkers kite marines; the marine counter is to rush in. About 30% that P qualifies.

## Amendments

1. **24 Nov, after the first calibration** (no payoff battle run yet): the first grid did not reach the 35-65%
   window. Z with 4 banelings and 8-16 zerglings gave Blue 0-7%; P with 3 zealots and 2-6 stalkers jumped from 85%
   (3 stalkers) to 23% (4). The grid is extended to vary both types:
   - Z: zerglings and banelings ∈ {(4,1), (6,1), (8,1), (10,1), (4,2), (6,2), (8,2)};
   - P: stalkers and zealots ∈ {(3,4), (3,5), (4,1), (4,2), (5,0), (4,0)}.

   The rule is unchanged: the setting closest to 50%.
2. **24 Nov, after the second calibration** (still no payoff battle): the outcomes jump steeply with one unit
   (Z 8+2 → 83%, 8+4 → 7%; P 4+2 → 84%, 4+3 → 23%). The grid is filled in between:
   - Z: {(8,3), (10,2), (12,2), (6,3)};
   - P: {(5,1), (5,2), (6,1), (3,3)}.

   The setting closest to 50% over all three rounds is used, whether or not it lies inside 35-65%. That is stated
   with the result.
