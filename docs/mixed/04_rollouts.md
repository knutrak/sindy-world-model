# Step 3: per-type closures and rollouts (exp08, 19 Nov)

Script: `experiments/mixed/rollouts.py`; model: `src/sindy_wm/models/typed_model.py`. Fitted on
exp08 `train`, scored on `val` (1000 battles). test3 not read.

## Team-level metrics (harness: health in ten marines' hp)

```
                 health_1s  health_2s  health_4s  health_8s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate
persistence          0.064      0.127      0.245      0.436        -0.033        -0.050     1.929        0.969                0.567       0.638          74.051          0.0
team A*-2k           0.022      0.037      0.068      0.121         0.354         0.053     0.793        0.607                0.136       0.648           4.990          0.0
typed (closure)      0.021      0.035      0.064      0.116         0.398         0.174     0.794        0.612                0.127       0.703           4.596          0.0
typed (rule)         0.021      0.035      0.064      0.115         0.433         0.227     0.800        0.612                0.125       0.709           4.542          0.0
```

Paired 95% bootstrap (by battle, 1000 resamples), model − team A*-2k, at 4 s:

```
                         health b-a          margin b-a
typed (closure)  [-0.0058, -0.0038]  [+0.0848, +0.1549]
typed (rule)     [-0.0058, -0.0040]  [+0.1409, +0.2073]
```

## Per-type metrics

Health per type in ten full units of the type; alive in units; composition = |error| of the
marauders' share of each side's alive units.

```
                 health_marine_2s  alive_marine_2s  health_marauder_2s  alive_marauder_2s  composition_2s  health_marine_4s  alive_marine_4s  health_marauder_4s  alive_marauder_4s  composition_4s
persistence                 0.051            0.688               0.027              0.293           0.077             0.099            1.353               0.053              0.575           0.132
typed (closure)             0.020            0.394               0.012              0.293           0.055             0.033            0.577               0.021              0.436           0.091
typed (rule)                0.023            0.430               0.013              0.304           0.061             0.036            0.614               0.021              0.454           0.099
```

## Closures (typed model)

```
               reach closure R² (count)           lam, beta, g allocation b0, b1, b2 (as attacker)
blue marine                    0.822719  [0.751, 0.691, 0.917]              [-0.011, 0.881, 1.788]
blue marauder                  0.940592  [1.191, 0.634, 1.023]               [-0.114, 0.93, 1.693]
red marine                     0.581582   [1.76, 0.572, 1.121]                 [0.73, 1.123, 2.24]
red marauder                   0.875799  [1.345, 0.579, 1.207]               [0.436, 1.035, 1.866]
```

## Typed model coefficients

```
                                                 typed (closure)
health_blue_marine: R[marine]*s[marine]                  -0.1702
health_blue_marine: R[marauder]*s[marine]                -0.1212
health_blue_marauder: R[marine]*s[marauder]              -0.0528
health_blue_marauder: R[marauder]*s[marauder]            -0.0811
health_red_marine: R[marine]*s[marine]                   -0.1732
health_red_marine: R[marauder]*s[marine]                 -0.1410
health_red_marine: R[marauder]*s[marine]*wd               0.1850
health_red_marauder: R[marine]*s[marauder]               -0.0547
health_red_marauder: R[marauder]*s[marauder]             -0.0926
health_red_marauder: R[marauder]*s[marauder]*wd           0.0974
alive_blue_marine: p                                      0.5309
alive_blue_marauder: p                                    0.4064
alive_red_marine: p                                       0.4533
alive_red_marine: focus                                   0.1260
alive_red_marauder: p                                     0.3729
alive_red_marauder: focus                                 0.0987
```

## Team model (A*-2k on team totals)

```
                              team A*-2k
health_blue: Rr                  -0.1775
health_red: Rb                   -0.1922
health_red: Rb*wd                 0.1710
distance: close*adv              -5.4331
distance: close*wd               -3.5966
distance: flee*wd                 0.4691
distance: close*hold*far         -2.9289
distance: close*hold*contact     -3.9937
alive_blue: 1                     0.6436
alive_red: 1                      0.5613
alive_red: focus                  0.2546
```
