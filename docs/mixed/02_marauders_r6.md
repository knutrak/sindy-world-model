# Step 1b: marauders alone (exp09_marauders_r6, reach 6)

Script: `experiments/mixed/marauders.py`. Scored on exp09 `val` (500 battles); `test` not read.
Health scale: 1 = ten full-health marauders (125 hp each). Game values: firepower
0.135 per marauder in reach (19 dmg per 1.125 s on 125 hp), walking speed 3.15.

## Prediction (harness metrics)

```
                           health_1s  health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate
persistence                    0.036      0.072      0.140        -0.003        -0.005     1.773        1.085                0.572       0.636          42.889          0.0
A, marines scaled              0.012      0.021      0.040         0.353         0.017     0.743        0.815                0.155       0.604           6.664          0.0
A*, marines scaled             0.012      0.020      0.039         0.365         0.033     0.748        0.815                0.148       0.624           6.249          0.0
A*-2k, marines scaled          0.012      0.020      0.039         0.357         0.030     0.747        0.815                0.149       0.626           6.302          0.0
A*-2k, refit on marauders      0.013      0.022      0.041         0.218        -0.022     0.746        0.639                0.154       0.640           6.002          0.0
```

## Coefficients (scaled marine models, and the refit)

```
                              A, marines scaled  A*, marines scaled  A*-2k, marines scaled  A*-2k, refit on marauders
health_blue: Rr                         -0.1150             -0.1182                -0.1150                    -0.1245
health_red: Rb                          -0.1210             -0.1182                -0.1210                    -0.1218
health_red: Rb*wd                        0.1061              0.1034                 0.1061                        NaN
distance: close*adv                     -6.0077             -6.0077                -6.0077                    -5.6136
distance: close*wd                      -3.1575             -3.1575                -3.1575                    -3.0493
distance: flee*wd                        0.2962              0.2962                 0.2962                     0.7317
distance: close*hold*far                -3.1830             -3.1830                -3.1830                    -3.4166
distance: close*hold*contact            -5.9147             -5.9147                -5.9147                    -5.2853
alive_blue: 1                            0.5175              0.6447                 0.6625                     0.6140
alive_red: 1                             0.5094              0.5520                 0.5394                     0.5397
alive_red: focus                         0.1355              0.2440                 0.2384                     0.1547
```
