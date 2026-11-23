# Unit-level geometry, stage U1: movement laws (results, 23 Nov 2026)

Plan: `13_unit_geometry_plan.md` (approved 23 Nov). Scripts: `experiments/generalisation/unit_geometry.py`
(rows, U1 fit) and `unit_geometry_diag.py` (failure diagnostic).

## U1: sparse per-unit movement laws

Target: each unit's velocity towards its nearest enemy (by reach margin) at the next step; library: margin terms
and gates, weapon readiness, danger, target in range, × Blue's intent; two-level STLSQ (0.05).

```
scenario side     type  rows (train)  terms  val R²  walking v∥ (m=3, advance)    U1
 marines blue   MARINE        526888     49   0.415                      2.543 False
 marines  red   MARINE        509590     12   0.154                      1.929 False
   mixed blue MARAUDER        457967     57   0.510                      2.449 False
   mixed blue   MARINE        676705     62   0.568                      2.589 False
   mixed  red MARAUDER        504282     11   0.134                      0.903 False
   mixed  red   MARINE        532553     11   0.152                      1.170 False
```

**U1 fails for every side and type** (≤ 6 terms, val R² ≥ 0.8 and walking speed within 15% of 3.15 asked). By the
plan, U2 and U3 are not run.

## Why: a diagnostic on 100 training battles per scenario

| | Blue | Red |
| --- | --- | --- |
| unit-steps at full speed (> 2.9) / stopped (< 0.3) / in between | marines 28 / 61 / 11%, mixed 21 / 64 / 15% | marines 42 / 41 / 17%, mixed 21 / 52 / 28% |
| moving at full speed: median cosine to the nearest enemy | −0.01 (marines), −0.50 (mixed) | 0.96 (marines), 0.98 (mixed) |
| moving at full speed: median cosine to own attack target | 0.99 (marines), −0.83 (mixed) | 0.98 |
| Red moving at full speed: median cosine to the map's attack point | | 0.16 / 0.12 |

Facts:
- When Red walks, it walks almost straight at its nearest enemy (the attack-point explanation offered in `12` is
  wrong).
- Movement is a switch, not a smooth function: units are mostly either at full speed (3.15) or stopped, with 11-28%
  of steps slowed or deflected by collision avoidance. When a unit switches depends on its own target and on crowding,
  not on its reach margin to the nearest enemy, so a sparse regression on velocity spreads the switch over dozens of
  terms and still explains only 13-57% of the variance.
- Blue's full-speed movement is often away from or across the nearest enemy (withdraw and kite in the training mix).

**Reading for next semester (not tested):** unit-level geometry would need a switching (hybrid) law — walk at speed v
towards the unit's own target unless it is in range — rather than one regression on velocity; the switch condition
and collision avoidance are the parts to model.
