# Three unit types: marines + marauders + stalkers (plan, 25 Nov 2026)

Asked for by Knut on 25 Nov ("experiment with three different units, somewhat similar; rigorous tests of the
pipeline"). Written **before** any exp12 data was collected. The expectations are Claude's; Knut has not written
his own. Every held-out set of the earlier work is spent and is not touched.

## Why these three

SMAClite's ranged, ground, single-target units are exactly marine, marauder and stalker. The rest are melee
(zealot, zergling), splash (baneling, colossus), a healer (medivac) or a structure (spine crawler).

| | hp + shield | armor | range | size | damage (bonus vs armored) | cooldown (ticks) | speed |
|---|---|---|---|---|---|---|---|
| marine | 45 | 0 | 5 | 0.75 | 6 | 10 | 3.15 |
| marauder (armored) | 125 | 1 | 6 | 1.125 | 10 (+10) | 18 | 3.15 |
| stalker (armored) | 80 + 80 | 1 | 6 | 1.25 | 13 (+5) | 22 | 4.13 |

Three things are new for the pipeline: a second walking speed; armor that does not apply to shield damage (so the
rate on a stalker depends on its shield); and shields that regenerate after 10 s without a hit, or from the start
when a battle starts damaged, so a health can rise.

## Data and model

`exp12_triple`: train 1500, val 500, test 500 battles, seeds 30 x00 000 +, collected from uncommitted code
(`allow_dirty`; the code is the pipeline commit 191b6ac plus the files of this plan). `TripleMixSampler`: one of four
mixes for both sides.

The frozen pipeline (`configs/generalisation_pipeline.yaml`), with one extension: proportional target choice for
three types (no new fitted numbers). With one or two types the code path and every fitted number are unchanged
(checked: refits of the three frozen models differ by 0.0). Withdraw radius 8, as exp08.

Game rates: damage per shot after armor / (cooldown × hp + shield of the target), with a stalker target's rate
between its shield phase (no armor) and its hp phase (armor 1); the check uses the whole unit, shots to kill.

Scenario flags (found when the first collection crashed at reset, before any battle): `ally_has_shields`,
`enemy_has_shields` and `num_unit_types` only shape SMAClite's *observations*, which divide by max shield and so
crash for a team that mixes shielded and unshielded units. They are off here; the shields themselves are simulated.

## Expectations (test, read once; each reported as met or not)

- **T1 structure:** every health equation keeps exactly its three firepower terms, no others, no wrong signs;
  distance at most 6 terms.
- **T2 physics:** each of the 18 firepower terms 0.75-1.0 × the game rate (the bounds of the earlier E2).
- **T3 prediction:** health error below persistence and Lanchester at 2 and 4 s (paired 95% intervals below 0).
- **T4:** below the team-level A*-2k at 2 s (paired).
- **T5:** reliable horizon (G6) at least 2 s.
- **T6:** health error at 2 s at most 1.2 × val.
- **T7 same numbers as before:** the 8 marine / marauder firepower terms within 10% of the frozen mixed model's.

Clarification (25 Nov, after the val run, before any test read): T1, T2 and T7 are judged on a refit on the test set
itself, as the earlier E1 / E2 were; T3-T6 on the train-fitted model. On val, with the train fit, T2 misses
(marine → marauder 1.014) and everything else is met; no bound is changed.

Walking speeds are reported, not judged: the types walk at two speeds and the distance equation has one.

Exploratory, val only: stalkers alone (exp07 train / val, reach capped at 6); the triple model unchanged on
marine-only, marauder-only, stalker-only and marine + marauder val battles (one model, several scenarios: RQ4).
