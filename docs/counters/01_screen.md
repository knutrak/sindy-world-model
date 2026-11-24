# Counter match-ups, stage 1: payoff screens

Plan: `00_plan.md` (approved 24 Nov). Script: `experiments/counters/screen.py`.

## Calibration (both sides advance/nearest, 100 battles per setting)

```
        won  timeout
tag                 
P2     1.00      0.0
P3     0.85      0.0
P3-3   0.85      0.0
P3-4   0.16      0.0
P3-5   0.00      0.0
P4     0.23      0.0
P4-0   1.00      0.0
P4-1   1.00      0.0
P4-2   0.84      0.0
P5     0.01      0.0
P5-0   0.99      0.0
P5-1   0.83      0.0
P5-2   0.28      0.0
P6     0.00      0.0
P6-1   0.31      0.0
Z10    0.03      0.0
Z10-1  0.97      0.0
Z10-2  0.57      0.0
Z12    0.01      0.0
Z12-2  0.38      0.0
Z14    0.00      0.0
Z16    0.00      0.0
Z4-1   1.00      0.0
Z4-2   1.00      0.0
Z6-1   1.00      0.0
Z6-2   0.94      0.0
Z6-3   0.76      0.0
Z8     0.07      0.0
Z8-1   1.00      0.0
Z8-2   0.83      0.0
Z8-3   0.40      0.0
```

Picked Red counts: {'Z': [10, 2], 'P': [6, 1]}.

## Z: Blue win rate (rows Blue, columns Red), 150 paired seeds per cell

```
red               advance/nearest  advance/focus  stand/nearest  withdraw/nearest  kite/nearest  spread/nearest
blue                                                                                                           
advance/nearest             0.553          0.853          1.000               1.0         0.467            1.00
advance/focus               0.993          1.000          1.000               1.0         0.987            1.00
stand/nearest               0.747          0.900          0.153               0.0         0.693            1.00
withdraw/nearest            0.033          0.080          0.000               0.0         0.013            0.66
kite/nearest                1.000          1.000          1.000               1.0         1.000            1.00
spread/nearest              1.000          1.000          1.000               1.0         1.000            1.00
```

Blue's best reply to each Red strategy:

```
red
advance/nearest        kite/nearest
advance/focus         advance/focus
stand/nearest       advance/nearest
withdraw/nearest    advance/nearest
kite/nearest           kite/nearest
spread/nearest      advance/nearest
```

Red's best reply to each Blue strategy:

```
blue
advance/nearest         kite/nearest
advance/focus           kite/nearest
stand/nearest       withdraw/nearest
withdraw/nearest       stand/nearest
kite/nearest         advance/nearest
spread/nearest       advance/nearest
```

## P: Blue win rate (rows Blue, columns Red), 150 paired seeds per cell

```
red               advance/nearest  advance/focus  stand/nearest  withdraw/nearest  kite/nearest  spread/nearest
blue                                                                                                           
advance/nearest             0.333          0.347          1.000              1.00         0.047           0.653
advance/focus               0.393          0.467          0.993              1.00         0.053           0.707
stand/nearest               0.027          0.013          0.007              0.00         0.000           0.233
withdraw/nearest            0.000          0.000          0.000              0.00         0.000           0.000
kite/nearest                0.773          0.853          1.000              1.00         0.140           1.000
spread/nearest              0.047          0.147          0.387              0.96         0.000           0.027
```

Blue's best reply to each Red strategy:

```
red
advance/nearest        kite/nearest
advance/focus          kite/nearest
stand/nearest       advance/nearest
withdraw/nearest    advance/nearest
kite/nearest           kite/nearest
spread/nearest         kite/nearest
```

Red's best reply to each Blue strategy:

```
blue
advance/nearest         kite/nearest
advance/focus           kite/nearest
stand/nearest       withdraw/nearest
withdraw/nearest     advance/nearest
kite/nearest            kite/nearest
spread/nearest          kite/nearest
```

## Read-outs

```
match-up best single Blue row  its mean                          near-dominant rows    P1  P2 headroom (points)     P2 95%  gate 1 (P1 and P2 >= 10)  timeouts
       Z         kite/nearest     1.000 advance/focus, kite/nearest, spread/nearest False                   0.0 [0.0, 0.0]                     False     0.109
       P         kite/nearest     0.794                                kite/nearest False                   0.0 [0.0, 0.0]                     False     0.112
```


## Verdict: gate 1 fails in both match-ups; the work stops at stage 1 (as the plan says)

- **Z (10 zerglings + 2 banelings):**
  - Blue kite, spread and advance/focus each win 99-100% of battles against every Red strategy.
  - The 57% calibration (advance vs advance) hid this: one Blue micro intent is enough to win outright.
  - The banelings' splash doesn't punish the clumped intents, because the marines kill them before they arrive.
  - Headroom is 0 points.
- **P (6 stalkers + 1 zealot; calibration 31%, outside the 35-65% window, picked as closest to 50%):**
  - Blue kite is the best reply to five of six Red strategies and ties the sixth (stand, 1.00).
  - It is near-dominant on its own (mean 0.79). Headroom is 0 points.
  - Red's kite is also Red's best reply to almost every Blue row.
  - Kite against kite gives Blue 14%: the stalkers' range (6 against the marines' 5) decides it.
- **Timeouts** (11% overall) occur only where both sides are passive (stand or withdraw on both sides). They count as Blue losses and change no best reply.
- **Reading:** the unit types change *who* wins, but not *which intent* wins. Kite is the best reply for the side that has it in every match-up tried so far. The intent menu has no move that beats kite, so no counter cycle appears.
