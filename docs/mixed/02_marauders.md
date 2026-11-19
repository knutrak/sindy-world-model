# Step 1b: marauders alone (exp09, 19 Nov)

Script: `experiments/mixed/marauders.py`. Scored on exp09 `val` (500 battles); `test` not read.
Health scale: 1 = ten full-health marauders (125 hp each). Game values: firepower
0.135 per marauder in reach (19 dmg per 1.125 s on 125 hp), walking speed 3.15.

## Prediction (harness metrics)

```
                           health_1s  health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate
persistence                    0.036      0.072      0.140        -0.003        -0.005     1.773        1.085                0.572       0.636          42.889          0.0
A, marines scaled              0.018      0.032      0.057        -0.024        -0.449     0.867        0.777                0.160       0.610           6.850          0.0
A*, marines scaled             0.018      0.031      0.055        -0.104        -0.585     0.963        0.777                0.153       0.622           6.560          0.0
A*-2k, marines scaled          0.017      0.031      0.054        -0.042        -0.491     0.956        0.777                0.154       0.624           6.553          0.0
A*-2k, refit on marauders      0.012      0.019      0.034         0.368         0.129     0.751        0.631                0.124       0.638           5.201          0.0
```

## Coefficients (scaled marine models, and the refit)

```
                              A, marines scaled  A*, marines scaled  A*-2k, marines scaled  A*-2k, refit on marauders
health_blue: Rr                         -0.1150             -0.1182                -0.1150                    -0.0771
health_red: Rb                          -0.1210             -0.1182                -0.1210                    -0.0864
health_red: Rb*wd                        0.1061              0.1034                 0.1061                     0.0778
distance: close*adv                     -6.0077             -6.0077                -6.0077                    -5.6136
distance: close*wd                      -3.1575             -3.1575                -3.1575                    -3.0493
distance: flee*wd                        0.2962              0.2962                 0.2962                     0.7317
distance: close*hold*far                -3.1830             -3.1830                -3.1830                    -2.9982
distance: close*hold*contact            -5.9147             -5.9147                -5.9147                    -3.8685
alive_blue: 1                            0.5175              0.6447                 0.6625                     0.6261
alive_red: 1                             0.5094              0.5520                 0.5394                     0.5329
alive_red: focus                         0.1355              0.2440                 0.2384                     0.1488
```
