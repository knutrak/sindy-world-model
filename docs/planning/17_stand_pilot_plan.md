# A hold-position intent: pilot before any recollection (21 Nov 2026)

**Status: written before the pilot is run (Knut asked for the intent and a pilot, 21 Nov). The decision rule
below is fixed now.**

## Why

In contact (about 91% of decision states), hold and advance play identical battles in 99% of cases: the scripts'
hold closes in as soon as any Blue unit has an enemy in range (`envs/intents.py`). The five candidates are three
distinct choices in contact. A **hold-position** intent that stays put in contact would be a fourth choice. The
question for the pilot: does it add decisions that matter, before any training data is recollected?

## The intent

`stand` (in `envs/intents.py`, outside `MOVEMENTS` / `INTENTS`, so every existing intent set is unchanged): Blue
units never move. Each unit attacks only enemies inside its own attack range, by the target rule (nearest, or the
team's focus target if it is in this unit's range, else the weakest in range); otherwise it stops. An attack order
on an enemy that SMAClite offers (within 6) but that is outside the unit's own range would make it walk, so those
are not given. Checked: in four S1 battles no Blue unit moved, and the units fired.

## The pilot (simulator only, no world model)

| | what | seeds | cost |
|---|---|---|---|
| P1 | **Twin check.** Battles played with random intents up to a drawn moment (0.5-8 s), then one of advance/nearest, advance/focus, stand/nearest, stand/focus for 4 s, then advance/nearest (as the counterfactual sets). Same seed, four branches | S1 and S2, 200 seeds each, 19 600 000 + / 19 610 000 + | minutes |
| P2 | **Fixed intents.** stand/nearest and stand/focus for the whole battle | Phase A's seeds (13 x00 000 +, 400 per scenario), paired with Phase A's fixed intents | minutes |
| P3 | **Oracle with seven candidates.** Phase A's exact oracle (re-plan 1 s, commit 2 s, horizon 8 s) with stand/nearest and stand/focus added to the five | Phase A's first 100 seeds of S1, S2, S3, S5, paired with Phase A's five-candidate exact oracle | about 30-40 min |

Before P3 is read: the five-candidate exact oracle is replayed on 10 of those seeds and must reproduce Phase A's
outcomes exactly (determinism check; otherwise P3 is paired against a fresh five-candidate run instead).

## Decision rule (fixed now)

A full recollection (new training data with `stand`, refit, tests) is worth it only if **both** hold:

- **(a) Distinct:** in contact at the branch, stand plays a different 4 s battle than advance with the same target
  in at least 50% of states (P1).
- **(b) It matters:** in at least one main scenario (S1, S2, S5), the seven-candidate oracle beats the five-candidate
  oracle by at least 3 win points with the paired 95% interval above 0 (P3), **or** a fixed stand intent beats
  Phase A's best fixed intent by at least 3 win points with the interval above 0 (P2).

S3 and S4 are reported, flagged as before. Also reported: how often the seven-candidate oracle plays stand.

## Expectations (written before)

(a) yes. (b) uncertain: standing gives up closing in, which costs fire when Red is out of range, but avoids units
walking into Red's fire; the most likely place for a gain is S3, where Blue outranges Red.

## Outputs

`experiments/planning/stand_pilot.py`; data in `data/processed/planning/stand_*.csv`; results in
`docs/planning/18_stand_pilot.md`.
