# Unit-level geometry with team-level damage: route 1 (plan, 23 Nov 2026)

**Status: approved by Knut 23 Nov 2026, as drafted; criteria fixed before any fit.** Exploratory groundwork for next semester's RQ5; the frozen pipeline is untouched.

## Why

Planning value in this testbed lies in timing, and the models miss it because they misjudge who is in range of whom
over the next seconds (`planning/45`, `generalisation/08`-`12`). Team-level summaries of geometry had no clean law.
SMAClite's movement is simple at unit level: a unit walks towards an enemy until it is in range, then stops and
fires. Withdraw and kite step back. So unit-level movement laws should be sparse, and reach can then be **counted**
instead of guessed.

**The question:** can SINDy discover sparse per-unit movement laws, and does a rollout that moves units by them
predict reach, and then the battle, better than the frozen closure?

## Data (all exists; per unit and step: position, velocity, cooldown, health, target; Blue's intent)

| scenario | fit | evaluate |
|---|---|---|
| marines | exp06 train (1,000 battles) | exp06 val |
| mixed | exp08 train + exp11 train (with kite; 1,000 + 500 battles) | exp08 val, exp11 val |

No held-out set is read, and nothing is collected.

## Stage U1: discover the movement laws

For each alive unit and step:
- **targets:** its velocity component **towards its nearest enemy** (v∥) and sideways (v⊥), from the recorded
  velocity;
- **unit features:** margin m (distance to the nearest enemy minus pair reach, as in `09`); gates Φ((c − m)/0.3) at
  fixed c ∈ {−1, −0.5, 0, 0.5, 1}; weapon readiness (cooldown / max cooldown); danger (nearest enemy within the
  withdraw radius); whether its current target is in range;
- **controls:** Blue's intent (advance, hold, withdraw, focus, kite, plus hold's contact flag); Red has no control
  (scripted).

Two-level STLSQ, as in `07`-`11`, separately for Blue and Red and per unit type.

**U1 passes** if, per side and type:
- v∥ keeps **at most 6 terms**;
- its val one-step R² is **at least 0.8**;
- the walking coefficient (v∥ for an advancing unit far from range) is **within 15% of the game's speed 3.15**.

v⊥ (collision avoidance) is reported, not ruled on.

## Stage U2: geometry alone

To isolate geometry from attrition, units are moved with the U1 laws from a recorded start, while units are removed
at their **recorded** death times. Reach is counted from the simulated positions, per side and type, as in the typed
tables.

**U2 passes** if the team reach error is **at most 0.7 × the frozen closure's** at 2 s **and** 4 s, in marines and
mixed (paired 95% intervals by battle). The measured-μ, σ mapping (`09`) is reported alongside.

## Stage U3: the full model

Positions follow the U1 laws. Damage is unit-level:
- each unit in range of an enemy fires at the target its side's rule picks: Blue by its intent's target rule,
  Red nearest;
- the rate per attacker type comes from the frozen pipeline's firepower k (no refit);
- a unit dies when its health reaches 0.

Rollouts start from recorded states, with Blue's recorded intents.

**U3 passes** if the margin error is **below the frozen model's** at 2 s and 4 s (paired intervals below 0) in
marines and mixed. Also reported: health and alive errors, and the win prediction over whole battles.

## What follows

- U3 passes: a planning test (counterfactual ranking, then closed loop with the matched ceiling) gets its own plan.
  This is the natural start of next semester's RQ5 work.
- U1 or U2 fails: reported, and the unit-level route is judged on why.

## Run time (estimated)

| step | time |
|---|---|
| code | 1-2 days |
| U1 fits | minutes |
| U2, U3 rollouts | minutes to tens of minutes (unit-level rollouts are heavier than team-level ones) |

## Expectations (written before any fit)

- **U1:** passes for Blue's advance and withdraw and for Red; kite and hold are harder. Walking coefficient
  3.0-3.2.
- **U2:** passes; reach error about 40-60% below the closure.
- **U3:** about 50%. The damage allocation (who shoots whom) is the uncertain part.

## Open questions for Knut

| | question | draft answer |
|---|---|---|
| Q1 | Unit-level damage in U3 (an interpretable, simplified simulator), or keep team-level damage fed by counted reach? | unit-level: deaths need to know *which* unit dies; team-level damage as a reported variant |
| Q2 | Is a model this close to the simulator's rules still "a world model discovered from data"? | yes, if the laws and rates are fitted, not copied; say so plainly in the text |
| Q3 | 1,000-1,500 training battles (subsampled), for speed? | yes |
