# Counters, step 2: a committed flank (plan, 24 Nov 2026)

**Status: on Knut's instruction of 24 Nov ("Try the split/flank with committed intents"). The read-out below is
fixed before any battle.**

## Why

Stage 1 (`01_screen.md`) failed gate 1 in both match-ups. Every intent on the menu is a way of fighting *well*: one
of them (kite, or with kite removed, spread or advance/focus) is best whatever Red does, so the payoff table has
no counters. A counter needs a **commitment with a trade-off**: a move that pays off against one opponent
behaviour and costs against another.

Split/flank against concentration is the classic one:
- **Flank should beat kite and stand.** A kiter steps away from its nearest enemy, which moves it into the other
  group. A standing ball gets caught in crossfire.
- **Concentration (advance) should beat flank.** While a split team walks around, the advancing team meets one half
  at a time: Lanchester's square law.
- **Kite beats advance**, as stage 1 showed.

If the game really behaves like this, kite > advance > flank > kite is a cycle, and no single intent dominates.

## The new intent: `flank` (committed)

At its first step, the team:
1. fixes its own centroid c and the enemy centroid e;
2. splits into two halves along the axis perpendicular to c → e (by projection, median split);
3. gives each half a waypoint beside the enemy: e ± F·p, with p the unit perpendicular and F = 7, clipped to
   the walkable bounds with a 1-unit margin.

**Transit.** Each unit walks to its half's waypoint. It doesn't react to the enemy moving, which is the commitment:
the waypoints are where the enemy *was*. The one exception is a unit with an enemy inside its own attack range: it
fights, as advance does.

**Attack.** A half switches to advance (the target rule as given) when its centroid is within 2.5 of its waypoint,
or after 10 s.

All other intents are unchanged. As in stage 1, every intent is played for the whole battle, so every table cell is
a commitment.

## Setting

- **Spawn distance 12-20** (was 4-16), so there is room to manoeuvre. The map is 31 × 15.
- **Menu (both sides):** advance/nearest, advance/focus, stand/nearest, kite/nearest, spread/nearest,
  flank/nearest. Withdraw is dropped: in stage 1 it only produced timeouts.
- **Match-ups:**
  - **M (mirror):** 7 marines + 3 marauders on both sides. It is symmetric, so it needs no calibration.
  - **Z:** Blue as before against 10 zerglings + 2 banelings (stage 1's calibration). The advance/advance cell is
    reported as the new calibration.
- **Seeds:** 150 paired seeds per cell, from 24 200 000 (M) and 24 300 000 (Z). Red's stream is seed + 7 000 000.
- **Size and run time:** 2 × 36 cells × 150 = 10 800 battles, about 15-20 min.
- **Smoke test first:** 20 battles, flank against advance in M, to check that the halves split and arrive.

## Read-out (fixed now, as in stage 1)

- **P1:** no Blue row within 5 points of the column best in every Red column.
- **P2:** headroom (the mean of the column bests − the best single row) ≥ 10 points, with a paired bootstrap 95%
  interval.
- **Gate 1 passes** for a match-up if P1 and P2 both hold.
- **Also reported:**
  - the best-reply structure, and whether it contains a cycle;
  - the same read-outs with flank removed. That isolates flank's contribution.

## Expectations (written before any battle)

- About 35% that at least one match-up passes.
- Most likely failure: flank loses to everything (split, walking units get caught). Second most likely: flank beats
  everything in M (crossfire), which would make it the new kite.

**If it passes:** stage 2 data collection, and the pipeline grows per-group states. **If it fails:** the counters
line stops for this semester, and the finding goes into RQ5's setup.

## Amendment 1 (before any battle counted)

The spawn sampler needs the minimum distance to fit at every bearing. On a 31 × 15 map, 12 doesn't fit. The
spawn distance becomes **9-16**: the largest minimum that fits, and still well above stage 1's 4.

## Amendment 2 (after a 20-battle smoke test, before the table)

In the smoke test, the halves never reached the 2.5 arrival radius. Every flank switched to attack only at the 10 s
timeout. There were two causes:
- Waypoints beside a Red team near the map edge get clipped close to it.
- A half crowds and stalls a few units short of its waypoint.

So a half now also attacks when it **stops getting closer**: less than 0.3 progress over the last 1 s. The arrival
radius becomes 3.5. The smoke test isn't part of the table (its seeds are 24 290 000+).

## Amendment 3 (the first table run failed before any result)

The spawn margin depends on the sampler's drawn unit count, and 8 of the 300 seeds didn't fit 9. The spawn distance
becomes **8.5-16**, which fits all 300 seeds. No battle result was seen.
