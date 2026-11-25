# The world-model interface (Phase 1, step 7)

Code: `src/sindy_wm/world_model.py`. Models: `configs/world_models/*.json`. Tests: `tests/test_world_model.py`.
Demo: `uv run scripts/world_model_demo.py` (or `... mixed 7`).

It wraps the frozen pipeline models (`docs/generalisation/06_frozen_equations.md`) without changing them.
Nothing was refitted or re-evaluated to make it.

## Use

```python
from sindy_wm.world_model import WorldModel

WorldModel.available()                    # ['marauders', 'marines', 'mixed']
wm = WorldModel.load("marines")
print(wm.equations())

state = wm.observe(env)                   # a live SMAClite game
state = wm.state_from_units(units)        # or a table of units from anywhere

pred = wm.predict(state, "hold/focus", horizon=2.0)
pred = wm.predict(state, [("withdraw/nearest", 1.0), ("advance/focus", 1.0)])
pred.margin_gain, pred.trajectory         # numbers: hp, units, map units, seconds
print(pred.to_text())                     # the same for a person or an LLM

wm.compare(state)                         # every intent, best predicted hp margin first
policy = wm.planner()                     # a Blue policy that plans with the model
```

`units` is a list of dicts or a DataFrame, one row per unit:

| column | meaning |
|---|---|
| `team` | `"blue"` (controlled) or `"red"` |
| `unit_type` | `"MARINE"` or `"MARAUDER"` |
| `x`, `y`, `hp` | centre position in map units, hit points |
| `shield`, `alive` | optional (default 0, and hp > 0) |

An intent is `<movement>/<target>`: `advance`, `hold` or `withdraw`, with `nearest` or `focus`. `kite`,
`stand` and the other later intents are refused: the frozen models were not fitted on them.

`compare` plays a single intent for 2 s and then `advance/nearest` to the horizon, as the planner experiments
did; `predict` plays exactly what it is given.

## How far to trust it

| model | reliable horizon | typical error per side at 2 s (hp, units, distance) |
|---|---|---|
| marines | 2 s | 12 hp, 0.56, 0.36 |
| marauders | 3 s | 25 hp, 0.49, 0.44 |
| mixed | 3 s | 15 hp, 0.50, 0.42 |

From the one held-out run (`docs/generalisation/04_test_results.md`). The reliable horizon is the longest with
margin R² of at least 0.3 and a lower interval bound above 0. Past it, predictions carry a warning.

- **Ranking intents is weaker than predicting outcomes.** Planners using the frozen model tied or lost to the
  best fixed intent in most scenarios and beat it only in S3 (`docs/planning/21_rq5_report.md`). In contact, hold
  and advance are nearly the same prediction, as they are nearly the same battle. Read small gaps in `compare`
  as ties.
- **Formation is held fixed** at its value in the starting state.
- **False stand-offs.** For shallow formations the closing gate can stop the teams just outside reach, and the
  model then predicts no damage at all. Such predictions carry a warning. On training battles (exploratory, 25 Nov:
  the first 600 battles per scenario, starts with no withdraw before the horizon), 3-4% of predictions at 2 and 3 s
  were stand-offs. The real battle fought on in 10-14% of them at 2 s and 18-35% at 3 s (0.3-0.6% and 0.6-1.2% of
  all predictions).
- **Not checked by the code:** whether the battle resembles the training battles (6-14 Red units, Blue / Red
  strength ratio 0.85-1.18 at the start, open map, the scripted Red AI). Uneven mixes were tested; underdog
  battles (S4) are outside the range.

## What ties it to SMAClite

Relevant for another simulator such as VR-Forces:

- unit hp, range and radius come from SMAClite's unit files (`states.typed.unit_stats`), also when a model is
  loaded from JSON; so SMAClite must be installed, and new unit types need an entry there and a refit;
- the firing limit of 6 and the withdraw radius are SMAClite and script settings, stored in each model's file;
- `state_from_units` is the only place a state enters, and takes plain unit tables.

## Rebuilding the files

```bash
uv run scripts/export_world_models.py
```

This needs the fitted pickle in `data/` (from `experiments/generalisation/frozen_models.py`). Loading the JSON
files does not.
