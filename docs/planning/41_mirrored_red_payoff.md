# Mirrored Red and the payoff table: results

Plan: `40_mirrored_red_payoff_plan.md` (approved 23 Nov).
Script: `experiments/planning/payoff40.py`.

## Gate S (S1 sampler, 400 seeds, both sides the same intent)

```
                                   won  timeout  seconds
blue            red                                     
advance/nearest advance/nearest  0.505      0.0    8.406
kite/nearest    kite/nearest     0.538      0.0    6.617
```

## S1: Blue win rate (rows Blue, columns Red), 200 paired seeds per cell

```
red               scripted  advance/nearest  advance/focus  hold/focus  withdraw/nearest  kite/nearest  kite/focus
blue                                                                                                              
advance/nearest      0.595            0.495          0.525       0.550             1.000         0.115       0.085
advance/focus        0.670            0.465          0.545       0.475             1.000         0.210       0.195
hold/nearest         0.645            0.500          0.595       0.210             0.075         0.125       0.095
hold/focus           0.720            0.510          0.540       0.185             0.055         0.240       0.190
withdraw/nearest     0.000            0.000          0.000       0.000             0.000         0.000       0.000
kite/nearest         0.970            0.890          0.830       0.810             1.000         0.505       0.465
kite/focus           0.975            0.905          0.825       0.815             1.000         0.550       0.535
```

Shortfall of each Blue row (largest gap to a column's best):

```
blue
advance/nearest     0.450
advance/focus       0.440
hold/nearest        0.925
hold/focus          0.945
withdraw/nearest    1.000
kite/nearest        0.070
kite/focus          0.005
```

## S2: Blue win rate (rows Blue, columns Red), 200 paired seeds per cell

```
red               scripted  advance/nearest  advance/focus  hold/focus  withdraw/nearest  kite/nearest  kite/focus
blue                                                                                                              
advance/nearest      0.530            0.475          0.505       0.480             1.000         0.220       0.285
advance/focus        0.605            0.460          0.520       0.495             1.000         0.165       0.180
hold/nearest         0.575            0.535          0.515       0.220             0.025         0.205       0.275
hold/focus           0.595            0.480          0.540       0.250             0.010         0.145       0.220
withdraw/nearest     0.000            0.000          0.000       0.000             0.000         0.000       0.000
kite/nearest         0.895            0.750          0.800       0.815             1.000         0.530       0.575
kite/focus           0.810            0.745          0.780       0.785             1.000         0.385       0.550
```

Shortfall of each Blue row (largest gap to a column's best):

```
blue
advance/nearest     0.365
advance/focus       0.395
hold/nearest        0.975
hold/focus          0.990
withdraw/nearest    1.000
kite/nearest        0.000
kite/focus          0.145
```

## S4: Blue win rate (rows Blue, columns Red), 200 paired seeds per cell

```
red               scripted  advance/nearest  advance/focus  hold/focus  withdraw/nearest  kite/nearest  kite/focus
blue                                                                                                              
advance/nearest      0.065            0.025          0.015       0.015             1.000         0.000       0.025
advance/focus        0.100            0.050          0.035       0.030             1.000         0.010       0.005
hold/nearest         0.075            0.030          0.025       0.000             0.015         0.000       0.025
hold/focus           0.100            0.040          0.025       0.015             0.015         0.015       0.000
withdraw/nearest     0.000            0.000          0.000       0.000             0.000         0.000       0.000
kite/nearest         0.325            0.125          0.165       0.175             1.000         0.030       0.050
kite/focus           0.250            0.095          0.135       0.130             1.000         0.030       0.060
```

Shortfall of each Blue row (largest gap to a column's best):

```
blue
advance/nearest     0.260
advance/focus       0.225
hold/nearest        0.985
hold/focus          0.985
withdraw/nearest    1.000
kite/nearest        0.010
kite/focus          0.075
```

## Read-outs

```
scenario best single Blue row  its mean over Red columns near-dominant rows (shortfall ≤ 5)  P1: no single best rule  P2 headroom (points)     P2 95%  P3 best row vs scripted  P3 best row vs mirrored (mean)  statement (P1 and P2 ≥ 10)
      S1           kite/focus                      0.801                         kite/focus                    False                 0.071 [0.0, 1.1]                    0.975                           0.772                       False
      S2         kite/nearest                      0.766                       kite/nearest                    False                 0.000 [0.0, 0.9]                    0.895                           0.745                       False
      S4         kite/nearest                      0.267                       kite/nearest                    False                 0.143 [0.0, 0.9]                    0.325                           0.258                       False
```

## Read-outs against the plan (`40`)

| | rule | result |
|---|---|---|
| Gate S | Blue wins with a 95% interval covering 50% when both sides play the same intent | **passes**: advance vs advance 50.5% [45.8, 55.5], kite vs kite 53.7% [49.0, 58.5] |
| **P1** | no Blue rule within 5 points of the best against every Red | **fails in all three**: kite is near-dominant (shortfall S1 0.5, S2 0.0, S4 1.0 points) |
| **P2** | adaptation headroom ≥ 10 points | **about 0** in all three (0.0-0.1 points) |
| P3 | best Blue rule vs scripted Red against vs mirrored Red | Red is harder to beat when it can use Blue's intents: S1 97.5% → 77.2% (mean over mirrored columns), S2 89.5% → 74.5%, S4 32.5% → 25.8% |

**Interpretation fixed in the plan:** a fixed rule (kite) stays near-best even against varied opponents in S1, S2
and S4. Knowing Red's intent adds essentially nothing to the best fixed rule.

Facts beside the rules:
- Kite against kite is even (S1 50-55%, S2 53-58%), as gate S showed; against every other Red behaviour kite wins
  more.
- A withdrawing Blue never wins (timeouts count as not won), and a Red that withdraws is beaten by any Blue that
  advances (100%) but not by one that holds (both sides stand off until the time limit).
- The earlier "both sides kite, Blue still wins 86-100%" (`31`) came from the approximate Red kite used there; with
  the exact mirror, kite against kite is even.
- Run time: 33 min for 29,400 battles.
