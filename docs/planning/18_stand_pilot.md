# Hold-position pilot: results

Plan and decision rule: `17_stand_pilot_plan.md`. Script: `experiments/planning/stand_pilot.py`. Simulator only.

## P1: is stand distinct? (counterfactual branches)

```
                                                                states  different 4 s battle  same winner  stand won  advance won
scenario pair                             at branch                                                                              
S1       stand/nearest vs advance/nearest in contact at branch     177                 0.734        0.847      0.452        0.605
                                          not in contact            16                 1.000        0.688      0.312        0.625
         stand/focus vs advance/focus     in contact at branch     177                 0.808        0.864      0.503        0.616
                                          not in contact            16                 1.000        0.688      0.375        0.562
S2       stand/nearest vs advance/nearest in contact at branch     183                 0.874        0.787      0.454        0.645
                                          not in contact            16                 1.000        0.875      0.625        0.625
         stand/focus vs advance/focus     in contact at branch     183                 0.902        0.820      0.508        0.678
                                          not in contact            16                 0.938        0.812      0.625        0.688
```

## P2: fixed stand intents against Phase A's best fixed intent (won, 400 paired seeds)

```
                          won best fixed (Phase A)  its won  difference               95%    n
scenario intent                                                                               
S1       stand/nearest  0.052           hold/focus    0.685      -0.632  [-0.680, -0.582]  400
         stand/focus    0.118           hold/focus    0.685      -0.568  [-0.618, -0.515]  400
S2       stand/nearest  0.020        advance/focus    0.602      -0.582  [-0.630, -0.535]  400
         stand/focus    0.072        advance/focus    0.602      -0.530  [-0.578, -0.485]  400
S3       stand/nearest  0.200           hold/focus    0.545      -0.345  [-0.393, -0.297]  400
         stand/focus    0.230           hold/focus    0.545      -0.315  [-0.362, -0.268]  400
S4       stand/nearest  0.002           hold/focus    0.105      -0.102  [-0.133, -0.075]  400
         stand/focus    0.005           hold/focus    0.105      -0.100  [-0.133, -0.070]  400
S5       stand/nearest  0.242        advance/focus    0.552      -0.310  [-0.355, -0.265]  400
         stand/focus    0.255        advance/focus    0.552      -0.298  [-0.343, -0.255]  400
```

## Determinism check (five-candidate oracle replayed on 10 seeds)

```
  scenario  seeds  same won  max |margin diff|
0       S1     10       1.0                0.0
1       S3     10       1.0                0.0
```

## P3: exact oracle with seven candidates against five (won, paired)

```
          oracle, 7 candidates  oracle, 5  difference               95%    n  stand/nearest  stand/focus
scenario                                                                                                
S1                        0.95       0.95        0.00  [+0.000, +0.000]  100          0.020        0.041
S2                        0.96       0.97       -0.01  [-0.050, +0.030]  100          0.044        0.050
S3                        0.98       0.98        0.00  [-0.030, +0.030]  100          0.041        0.024
S5                        0.83       0.82        0.01  [-0.030, +0.050]  100          0.044        0.035
```

