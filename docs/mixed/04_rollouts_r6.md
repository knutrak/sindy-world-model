# Step 3: per-type closures and rollouts (exp08, 19 Nov)

Script: `experiments/mixed/rollouts.py`; model: `src/sindy_wm/models/typed_model.py`. Fitted on
exp08 `train`, scored on `val` (1000 battles). test3 not read.

## Team-level metrics (harness: health in ten marines' hp)

```
                           health_1s  health_2s  health_4s  health_8s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate
persistence                    0.064      0.127      0.245      0.436        -0.033        -0.050     1.929        0.969                0.567       0.638          74.051          0.0
team A*-2k                     0.021      0.039      0.072      0.122         0.335         0.039     0.771        0.609                0.141       0.671           5.256          0.0
typed (closure)                0.022      0.038      0.072      0.129         0.265        -0.017     0.810        0.608                0.151       0.718           5.528          0.0
typed (rule)                   0.022      0.038      0.072      0.126         0.258         0.015     0.788        0.608                0.146       0.730           5.584          0.0
typed (closure) + wd rule      0.020      0.034      0.065      0.121         0.370         0.101     0.787        0.608                0.139       0.663           4.975          0.0
typed (rule) + wd rule         0.020      0.035      0.065      0.116         0.381         0.149     0.776        0.608                0.136       0.661           4.956          0.0
```

Paired 95% bootstrap (by battle, 1000 resamples), model − team A*-2k, at 4 s:

```
                                   health b-a          margin b-a
typed (closure)            [-0.0008, +0.0013]  [-0.1028, -0.0110]
typed (rule)               [-0.0010, +0.0009]  [-0.0723, +0.0211]
typed (closure) + wd rule  [-0.0073, -0.0053]  [+0.0325, +0.0941]
typed (rule) + wd rule     [-0.0081, -0.0063]  [+0.0834, +0.1411]
```

## Per-type metrics

Health per type in ten full units of the type; alive in units; composition = |error| of the
marauders' share of each side's alive units.

```
                           health_marine_2s  alive_marine_2s  health_marauder_2s  alive_marauder_2s  composition_2s  health_marine_4s  alive_marine_4s  health_marauder_4s  alive_marauder_4s  composition_4s
persistence                           0.051            0.688               0.027              0.293           0.077             0.099            1.353               0.053              0.575           0.132
typed (closure)                       0.021            0.394               0.013              0.290           0.056             0.035            0.588               0.023              0.434           0.093
typed (rule)                          0.024            0.424               0.014              0.301           0.061             0.037            0.598               0.023              0.452           0.097
typed (closure) + wd rule             0.020            0.382               0.012              0.282           0.055             0.033            0.574               0.021              0.422           0.091
typed (rule) + wd rule                0.022            0.415               0.013              0.293           0.060             0.035            0.596               0.022              0.439           0.097
```

## Closures (typed model)

```
               reach closure R² (count)           lam, beta, g allocation b0, b1, b2 (as attacker)
blue marine                    0.822719  [0.751, 0.691, 0.917]              [-0.011, 0.881, 1.788]
blue marauder                  0.836237  [1.015, 0.695, 1.037]               [-0.114, 0.93, 1.693]
red marine                     0.581582   [1.76, 0.572, 1.121]                 [0.73, 1.123, 2.24]
red marauder                   0.738353  [1.211, 0.441, 1.258]               [0.436, 1.035, 1.866]
```

## Typed model coefficients

```
                                               typed (closure)
health_blue_marine: R[marine]*s[marine]                -0.1784
health_blue_marine: R[marauder]*s[marine]              -0.1651
health_blue_marauder: R[marine]*s[marauder]            -0.0600
health_blue_marauder: R[marauder]*s[marauder]          -0.1127
health_red_marine: R[marine]*s[marine]                 -0.1845
health_red_marine: R[marauder]*s[marine]               -0.1626
health_red_marauder: R[marine]*s[marauder]             -0.0596
health_red_marauder: R[marauder]*s[marauder]           -0.1125
alive_blue_marine: p                                    0.5213
alive_blue_marauder: p                                  0.3925
alive_red_marine: p                                     0.4472
alive_red_marine: focus                                 0.1236
alive_red_marauder: p                                   0.3496
alive_red_marauder: focus                               0.0882
```

## Team model (A*-2k on team totals)

```
                              team A*-2k
health_blue: Rr                  -0.2177
health_red: Rb                   -0.2195
health_red: Rb*wd                 0.1855
distance: close*adv              -5.4331
distance: close*wd               -3.5966
distance: flee*wd                 0.4691
distance: close*hold*far         -3.0806
distance: close*hold*contact     -5.2624
alive_blue: 1                     0.6391
alive_red: 1                      0.5521
alive_red: focus                  0.2557
```
