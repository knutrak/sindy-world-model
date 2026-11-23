# A richer decision space: per-type roles and target priority: results

Plan: `48_richer_intents_plan.md` (approved 23 Nov). Script: `experiments/planning/richer48.py`.
`34`'s first 200 seeds per scenario, paired.

## Room for a planner

```
scenario             best fixed of 27  its wins  oracle27-h3 wins  Room27       Room27 95% best fixed team intent  its wins   oracle5-h3 wins  Room5        Room5 95%  N1 = Room27 − Room5  N3 oracle27 − oracle5           N3 95%
      S2 role:advance-advance:nearest      0.58              0.89    0.31 [+0.240, +0.375]    fixed:advance/focus      0.615            0.915   0.30 [+0.235, +0.365]                 0.01                 -0.025 [-0.070, +0.020]
      S4 role:advance-advance:nearest      0.07              0.33    0.26 [+0.200, +0.325]       fixed:hold/focus      0.065            0.285   0.22 [+0.165, +0.280]                 0.04                  0.045 [-0.015, +0.110]
      S5 role:advance-advance:nearest      0.56              0.63    0.07 [+0.035, +0.105]    fixed:advance/focus      0.545            0.645   0.10 [+0.060, +0.145]                -0.03                 -0.015 [-0.050, +0.015]
```

**Statement (N1 ≥ 5 points in at least two of S2, S4, S5): does not hold.**

**N2 (one fixed combination within 5 points of the best in all three scenarios):** role:advance-advance:marine, role:advance-advance:nearest.

## Fixed combinations (wins)

```
                                    S2     S4     S5
policy                                              
role:advance-advance:nearest     0.580  0.070  0.560
role:advance-advance:marine      0.565  0.055  0.550
role:advance-advance:marauder    0.510  0.060  0.555
role:advance-stand:marauder      0.180  0.010  0.305
role:advance-stand:marine        0.180  0.010  0.235
role:advance-stand:nearest       0.170  0.005  0.230
role:stand-advance:marauder      0.120  0.000  0.410
role:stand-advance:marine        0.110  0.000  0.335
role:stand-advance:nearest       0.110  0.005  0.360
role:stand-stand:marauder        0.070  0.000  0.300
role:stand-stand:marine          0.060  0.000  0.200
role:stand-stand:nearest         0.045  0.000  0.230
role:advance-withdraw:marine     0.025  0.000  0.000
role:advance-withdraw:nearest    0.010  0.000  0.000
role:advance-withdraw:marauder   0.005  0.000  0.000
role:withdraw-advance:marauder   0.005  0.000  0.275
role:withdraw-advance:nearest    0.005  0.000  0.235
role:stand-withdraw:marauder     0.000  0.000  0.000
role:stand-withdraw:marine       0.000  0.000  0.000
role:stand-withdraw:nearest      0.000  0.000  0.000
role:withdraw-advance:marine     0.000  0.000  0.170
role:withdraw-stand:marauder     0.000  0.000  0.080
role:withdraw-stand:marine       0.000  0.000  0.035
role:withdraw-stand:nearest      0.000  0.000  0.060
role:withdraw-withdraw:marauder  0.000  0.000  0.000
role:withdraw-withdraw:marine    0.000  0.000  0.000
role:withdraw-withdraw:nearest   0.000  0.000  0.000
```

## Read-outs against the plan (`48`)

| | rule | result |
|---|---|---|
| Room27 | `oracle27-h3` − best of 27 fixed | S2 +31, S4 +26, S5 +7 points (all intervals above 0) |
| Room5 | `oracle5-h3` − best fixed team intent | S2 +30, S4 +22, S5 +10 |
| **N1** | Room27 − Room5 ≥ 5 in two of three | **+1, +4, −3: does not hold** |
| N2 | one fixed combination near-best everywhere | yes: both types advance, target nearest (or marines first) |
| N3 | oracle27 − oracle5 | ties in all three (−2.5, +4.5, −1.5) |

**Statement: does not hold.** Per-type roles and target-type priority do not give a planner more room than the five
team-level intents; even a perfect model gains nothing from the extra choices.

Facts beside the rules:
- The best fixed combination is the same everywhere: both types advance, nearest target. Every combination with a
  standing or withdrawing type is far worse as a fixed rule (standing marines and marauders together: S2 4.5-7%), so
  "a real hold" is not a good *fixed* choice; target-type priority changes little (±1-7 points).
- The room for a planner is in *when* to switch between a few team-level behaviours, not in a larger menu.
- Run time: 1 h 41 min for 16,800 battles (the 27-candidate oracle about 13 s per battle); estimated 40-45 min.
