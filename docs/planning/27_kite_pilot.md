# Kite pilot: results

Plan and decision rule: `26_kite_pilot_plan.md`. Script: `experiments/planning/kite_pilot.py`. Simulator only.

## P1: is kite distinct? (counterfactual branches)

```
                                                               states  different 4 s battle  same winner  kite won  advance won
scenario pair                            at branch                                                                             
S1       kite/nearest vs advance/nearest in contact at branch     156                 0.968        0.737     0.769        0.506
                                         not in contact            17                 1.000        0.706     0.941        0.647
         kite/focus vs advance/focus     in contact at branch     156                 0.955        0.712     0.763        0.500
                                         not in contact            17                 1.000        0.647     0.941        0.588
S2       kite/nearest vs advance/nearest in contact at branch     189                 0.947        0.825     0.735        0.603
                                         not in contact             9                 0.889        0.889     0.889        0.778
         kite/focus vs advance/focus     in contact at branch     189                 0.958        0.868     0.677        0.619
                                         not in contact             9                 0.889        1.000     0.889        0.889
```

## P2: fixed kite intents against Phase A's best fixed intent (won, 400 paired seeds)

```
                         won best fixed (Phase A)  its won  difference               95%    n
scenario intent                                                                              
S1       kite/nearest  0.985           hold/focus    0.685       0.300  [+0.255, +0.347]  400
         kite/focus    0.958           hold/focus    0.685       0.272  [+0.228, +0.318]  400
S2       kite/nearest  0.885        advance/focus    0.602       0.282  [+0.233, +0.330]  400
         kite/focus    0.855        advance/focus    0.602       0.252  [+0.207, +0.297]  400
S3       kite/nearest  0.985           hold/focus    0.545       0.440  [+0.393, +0.487]  400
         kite/focus    0.990           hold/focus    0.545       0.445  [+0.395, +0.492]  400
S4       kite/nearest  0.340           hold/focus    0.105       0.235  [+0.190, +0.278]  400
         kite/focus    0.242           hold/focus    0.105       0.138  [+0.098, +0.180]  400
S5       kite/nearest  0.590        advance/focus    0.552       0.038  [+0.013, +0.065]  400
         kite/focus    0.575        advance/focus    0.552       0.022  [+0.000, +0.045]  400
```

## Determinism check (five-candidate oracle replayed on 10 seeds)

```
  scenario  seeds  same won  max |margin diff|
0       S1     10       1.0                0.0
1       S3     10       1.0                0.0
```

## P3: exact oracle with seven candidates against five (won, paired)

```
          oracle, 7 candidates  oracle, 5  difference               95%    n  kite/nearest  kite/focus
scenario                                                                                              
S4                        0.79       0.67        0.12  [+0.040, +0.200]  100         0.226       0.146
S5                        0.83       0.82        0.01  [-0.030, +0.050]  100         0.265       0.139
```

