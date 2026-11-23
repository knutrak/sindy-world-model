# A Red with Blue's intents, and the payoff table (plan, 23 Nov 2026)

**Status: approved by Knut 23 Nov 2026, as drafted; rules fixed before any battle.**

## Why

All results so far are against one scripted Red that always attack-moves. The oracle's key move (a short step back
at first contact while Red is strung out) and the dominance of kite may be properties of that opponent.

**The question:** if Red can use the same intents as Blue, is there still one Blue rule that is (nearly) best against
every Red, or does the best Blue choice depend on what Red does? Only in the second case does a planner have
something to adapt to that a fixed rule can't.

No world model is involved. This is a simulator-only screen.

## The mirrored Red (code)

A thin mirrored view of the game:
- own units = Red, enemies = Blue;
- each Red unit's action mask comes from SMAClite's own mask function with the Blue units as targets (same
  walkable-terrain check, same firing limit 6 as Blue).

Blue's existing `IntentPolicy` then chooses Red's actions from this view. They are turned into Red commands by
SMAClite's own rules:

| action | Red command |
|---|---|
| stop | StopCommand |
| move | MoveCommand to 2 units in that direction |
| attack unit k | AttackUnitCommand on Blue unit k |

Red decides at the same steps as Blue.

**One stated change:** the scripted Red acquires targets at 5.375; a mirrored Red has Blue's rules (6). The scripted
attack-move stays in the table as its own column, so everything links back to earlier results.

**Tests** (in the test suite):
- with no mirrored Red, battles are unchanged;
- the mirrored mask lets a Red unit attack exactly the Blue units within its reach;
- a mirrored Red with a stop intent never moves.

## Gate S (symmetry, before the payoff battles)

Both sides play advance/nearest (mirrored Red) on the S1 sampler (marines; strength ratios drawn symmetrically around
1), with 400 fresh seeds. **Blue's win rate must have a 95% interval covering 50%.** The same is reported for both
sides playing kite/nearest. If the gate fails, the payoff screen does not run, and the asymmetry is investigated and
reported.

## The payoff table

| | |
|---|---|
| Blue rows (7, the planner's candidates) | advance/nearest, advance/focus, hold/nearest, hold/focus, withdraw/nearest, kite/nearest, kite/focus |
| Red columns (7) | **scripted attack-move**, and mirrored advance/nearest, advance/focus, hold/focus, withdraw/nearest, kite/nearest, kite/focus |
| scenarios | S1 (marines, balanced), S2 (mixed, balanced), S4 (mixed, Blue the underdog) |
| battles | 200 per cell, on the same fresh seeds in every cell (paired): S1 22 000 000 +, S2 22 100 000 +, S4 22 300 000 + |

That is 49 cells × 200 × 3 = 29,400 battles. Cells with a withdrawing side run long (up to the 60-90 s limit). The
estimate is 1-2 h on 8 workers, timed on a small run first.

## Read-outs and rules (fixed once approved)

| | what | how |
|---|---|---|
| **P1 dominance** | per scenario, each Blue row's **shortfall** = its largest gap, over Red columns, to that column's best Blue row (win points) | a row with shortfall ≤ 5 points is "near-dominant". **"No single Blue rule is best against every Red" holds in a scenario if no row is near-dominant** |
| **P2 adaptation headroom** | mean over Red columns of the column-best Blue win rate, minus the best single row's mean | in win points: what knowing Red's intent would add over the best fixed rule |
| P3 harder to beat? | the best Blue row's win rate against the mirrored-Red columns vs against the scripted attack-move | reported |

**Interpretation, fixed now:**
- P1 holds **and** P2 ≥ 10 points in a scenario: planning against varied opponents has clear room there.
- Otherwise, a fixed rule stays (nearly) best even against varied opponents.

Either way the result is a description of the testbed, not a planner test.

## Expectations (written before any battle)

- **Gate S** passes.
- **P1** holds in S1 and S2: kite loses its edge against a kiting or holding Red. In S4 it is uncertain.
- **P2:** 5-15 points.
- **P3:** yes, Red is clearly harder to beat when it can kite.

## Run time

| step | time |
|---|---|
| code and tests | about half a day |
| gate S | minutes |
| payoff table | 1-2 h |

## Open questions for Knut

| | question | draft answer |
|---|---|---|
| Q1 | Include S1, or only the mixed scenarios? | include: marines are the simplest symmetric case |
| Q2 | 200 battles per cell (about ±3.5 points standard error)? | yes; 49 cells × 3 scenarios is already 29,400 battles |
| Q3 | The 5-point margin for "near-dominant"? | yes: about 1.5 standard errors of a cell |
