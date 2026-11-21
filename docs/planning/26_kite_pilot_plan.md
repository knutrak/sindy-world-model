# A kite intent: pilot before any recollection (22 Nov 2026, overnight run)

**Status: written before the pilot is run; the decision rule is fixed now. A first look at 6 battles (3 in S1,
3 in S3-type battles, seeds 0-2) was taken while checking the intent works: kite won 6, advance 4.**

## Why

The exact oracle's main deviation from advance is a short step back at first contact (`19`); with five candidates
it can only build kiting from 1 s withdraw / advance steps. A per-unit kite does it at the weapon's own rhythm.

## The intent

`kite` (in `envs/intents.py`, outside `MOVEMENTS` / `INTENTS`): with an enemy inside the unit's own attack range, the
unit shoots when its weapon is ready (cooldown ≤ one step, by the target rule as `stand`) and steps away from the
nearest enemy while it reloads; with nothing in range it advances (as advance).

## The pilot (simulator only; the same design as `17`)

| | what | seeds |
|---|---|---|
| P1 | twin check: early branches advance/nearest, advance/focus, kite/nearest, kite/focus for 4 s, then advance/nearest | S1, S2, 200 each (19 620 000 + / 19 630 000 +) |
| P2 | fixed kite/nearest and kite/focus for the whole battle | Phase A's seeds, 400 per scenario, paired with Phase A's fixed intents |
| P3 | exact oracle with seven candidates (the five + kite/nearest, kite/focus) | Phase A's first 100 seeds of **S4 and S5** (the oracle is at 95-98% in S1-S3, so a gain cannot show there), paired with Phase A's exact oracle |

## Decision rule (fixed now)

A full recollection is worth it only if (a) in contact, kite plays a different 4 s battle than advance with the same
target in at least 50% of states (P1), **and** (b) in at least one main scenario (S1, S2, S5), a fixed kite intent
beats Phase A's best fixed intent by at least 3 win points with the paired 95% interval above 0 (P2), or the
seven-candidate oracle beats the five-candidate one by at least 3 points with the interval above 0 in S5 (P3).
S3 and S4 are reported, flagged.

## Expectations

(a) yes. (b): P2 plausible from the first look (it may raise the baseline rather than show planning value); P3 small.
