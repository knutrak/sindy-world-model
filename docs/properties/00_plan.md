# Unit properties, one at a time (plan, 25 Nov 2026)

**Status: drafted by Claude; rules approved by Knut 25 Nov 2026 as drafted. Nothing has been collected or fitted.** The only check made: a custom
marine (cooldown 1.25) loads in SMAClite. No battle was played.

## Why

RQ4 asks about scenarios with other "unit properties". In our data, unit types differ in hp, damage, cooldown, armour,
range and speed together, so no single property's effect can be separated (caveat in `rq4_applicability`). The claim
under test:

> The frozen marine model carries over to a unit with one changed property when its firepower is rescaled by the
> game's rate, without refitting.

If it holds, firepower has a rule in terms of unit properties, i.e. one model with parameters (project description 4.5).

## The model under test (frozen, unchanged)

```
H_B' = -0.1770·R_R          H_R' = -0.1861·R_B      (R = enemy units in reach; Blue's × (1 - wd))
(ln A_B)' = 0.662·(ln H_B)' (ln A_R)' = (0.562 + 0.260·focus)·(ln H_R)'
d' = -0.485·walk_r·wd + 0.575·danger·wd - 6.008·close·adv - 3.183·close·hold·far - 5.915·close·hold·contact
```

Game rate of a firing marine: 1 / (hp / (damage - armour) × cooldown) = 1 / (7.5 shots × 0.625 s) = 0.2133 per s.

**Rescaling rule (fixed now):** k_new = k_marine × rate_new / rate_marine. Speed variants: all five distance
coefficients × speed_new / 3.15. Range variants: ρ = range + 0.75. Nothing else changes (p, gates, reach factors, withdraw
radius 7.0).

## Variants (both sides have the same unit; one property changed)

| variant | change | rate ratio | predicted k_B / k_R |
|---|---|---|---|
| V0 | none (custom copy of the marine) | 1 | 0.177 / 0.186 |
| D−, D+ | damage 3, 12 (from 6) | 0.50, 2.00 | 0.089 / 0.093, 0.354 / 0.372 |
| C−, C+ | cooldown 6, 20 ticks (from 10): 0.375, 1.25 s | 1.67, 0.50 | 0.295 / 0.310, 0.089 / 0.093 |
| H−, H+ | hp 30, 90 (from 45) | 1.50, 0.50 | 0.266 / 0.279, 0.089 / 0.093 |
| A1, A3 | armour 1, 3 (from 0) | 0.83, 0.50 | 0.148 / 0.155, 0.089 / 0.093 |
| S−, S+ | speed 2.1, 4.725 (from 3.15) | 1 | unchanged; distance × 0.67, 1.50 |
| R4, R3 | range 4, 3 (from 5); more is not possible, reach is capped at 6 | 1 | unchanged; ρ = 4.75, 3.75 |

D−, A3, H+ and C+ share the predicted ratio 0.50 by four different routes, a check that the rule is about the rate and
not about one property. Size is not varied (it changes collisions and formations too).

## Data

`exp13_properties`: the sampler, intents, switching schedule and step (0.125 s) of exp06; seeds 31 x00 000 + (unused).
13 sets × 1000 battles; per set 500 for the refit and 500 for scoring, split by battle. Committed code only.

**Run time (estimate, to be confirmed by a timing pilot of 20 battles per variant, not analysed):** exp06 took 6.3 min
per 1000 battles on one core; battles with half the firepower last longer. About 13,000 battles: 1.5-3 core hours, so
25-45 min on 4 workers. State building and fits: unknown, stated before they start.

## Models compared on each set's 500 scoring battles

1. persistence;
2. the frozen marine model, unchanged (for R4 / R3 with the old ρ);
3. the frozen model **rescaled** by the rule above;
4. the pipeline **refitted** on the set's other 500 battles (upper reference).

## Expectations

**Knut's, written before approval:** none. Knut approved the rules on 25 Nov and chose to run without writing an
expectation; the lines below were left blank. The rules P1-P5 are Claude's proposals.

- coefficients: …
- prediction: …
- which variant is most likely to break the rule, and why: …

**Rules (proposed; change them before approval, not after):**

- **G0 (check, must pass before anything else):** V0 gives exactly the same battles as the built-in marine on 20 seeds.
- **P1 firepower:** in the refits of D, C, H, A (8 variants, 2 sides), k_refit / k_refit(V0) is within ±15% of the rate
  ratio. Met if at least 14 of 16.
- **P2 no effect:** in S and R (4 variants, 2 sides) the same ratio is within ±15% of 1. Met if at least 7 of 8.
- **P3 speed:** in S−, S+ the advance and hold-far coefficients scale with speed within ±20%.
- **P4 prediction:** in D, C, H, A the rescaled model's health error at 2 s is below the unchanged model's (paired 95%
  interval below 0). Met if in all 8.
- **P5 cost of not refitting:** in D, C, H, A the rescaled error at 2 s is at most 1.15 × the refit's. Met if in 7 of 8.

Reported, not judged: errors at 0.5, 1 and 4 s, margin R², alive and distance errors, kept terms, all S and R predictions.

**Reading, fixed now:** P1 + P4 + P5 met: firepower follows the rule. P1 met, P5 not: the coefficients follow it, but
other parts (alive law, reach closure) do not carry over. P1 not met: the rule fails, and RQ4 is limited accordingly;
the failing properties are named.

## How it could fail (known mechanisms, not predictions)

- Firing share: k is about 0.83-0.87 × the game rate, and that share may depend on the cooldown.
- Overkill: with few shots per kill, more damage is wasted.
- The rates' smoothing window (9 steps, 1.1 s) is short against a 1.25 s cooldown.
- Very fast or slow battles change how much happens in 2 s; hence the comparison with the refit on the same set.
- Red's AI picks up targets within scan range 5 + target radius, whatever the weapon range.

## Code needed

- unit files `configs/units/*.json` and 13 scenario files (SMAClite reads `custom_unit_path`, relative to the working
  directory: make it absolute in `make_env`);
- `smaclite_snapshot.py:45`: custom units have no `.name`; record their stats' name;
- `states/typed.py` `unit_stats`: look up custom units as well as built-in ones;
- one collection script, one analysis script. **The pipeline itself is not changed**; the frozen refits must still differ
  by 0.0 (existing check).

## Stopping rules

Run once. No new variants, thresholds or pipeline changes after the results are seen; a failure is reported as it is.
Not part of this plan: discrete-time models, melee or support units, mixed armies with changed units, planning.

## Implementation notes (25 Nov, before any exp13 battle)

- Custom units are registered by name from `configs/units/` (`src/sindy_wm/envs/units.py`), so `custom_unit_path` is not
  used. All variants use the same 1000 seeds (31 000 000 +); G0 uses 31 900 000 +, timing 31 800 000 +.
- Scripts: `scripts/collect_properties.py`, `experiments/properties/properties.py` (g0, build, analyse).
- Checks: 239 tests pass; refits of the frozen marine, marauder and mixed models differ by 0.0.
- Dry run of `analyse` on exp06 val relabelled as every variant (no new data): 35 s. As it should, the rescaled models fail
  on unchanged marines (P1 0 of 16, P4 0 of 8) and V0 equals the frozen model. One check was too strict (rounding), fixed.
