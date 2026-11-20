# Withdraw movement: fitted gates or the scripts' rule (val, 21 Nov)

Script: `experiments/generalisation/withdraw_movement.py`. Firing under withdraw: the frozen rule;
target choice: proportional. Distance in map units; health in ten units of the first type.

## marines (withdraw radius 7.00)

Start points withdrawing > 50% of the first 2 s: 1850 of 8946.

```
       distance_1s  distance_2s  distance_4s  distance_1s, withdrawing  distance_2s, withdrawing  distance_4s, withdrawing  health_2s  health_4s  margin_r2_2s  battle_health_nrmse  duration_err_s
gates       0.2459       0.3490       0.4925                    0.4293                    0.5965                    0.6832     0.0279     0.0534        0.3818               0.1342          3.2198
rule        0.2576       0.3594       0.4986                    0.4789                    0.6512                    0.7094     0.0277     0.0523        0.4012               0.1316          2.8278
```

Distance equation (train, full):

- gates: {'close*adv': -6.008, 'close*wd': -3.157, 'flee*wd': 0.296, 'close*hold*far': -3.183, 'close*hold*contact': -5.915}
- rule: {'walk_r*wd': -0.485, 'danger*wd': 0.575, 'close*adv': -6.008, 'close*hold*far': -3.183, 'close*hold*contact': -5.915}

Withdraw coefficients across refits:

```
                              close*wd  flee*wd  walk_r*wd  danger*wd
gates train (full)              -3.157    0.296        NaN        NaN
      val (refit)               -3.163    0.275        NaN        NaN
      train, 500 battles (a)    -3.140    0.293        NaN        NaN
      train, 500 battles (b)    -3.138    0.260        NaN        NaN
rule  train (full)                 NaN      NaN     -0.485      0.575
      val (refit)                  NaN      NaN     -0.538      0.589
      train, 500 battles (a)       NaN      NaN     -0.530      0.606
      train, 500 battles (b)       NaN      NaN     -0.556      0.612
```

## marauders (withdraw radius 8.67)

Start points withdrawing > 50% of the first 2 s: 1641 of 7832.

```
       distance_1s  distance_2s  distance_4s  distance_1s, withdrawing  distance_2s, withdrawing  distance_4s, withdrawing  health_2s  health_4s  margin_r2_2s  battle_health_nrmse  duration_err_s
gates       0.3192       0.4672       0.6394                    0.6452                    0.8698                    0.9130     0.0203     0.0386        0.3437               0.1393          6.3122
rule        0.3179       0.4666       0.6374                    0.6353                    0.8728                    0.9104     0.0203     0.0386        0.3436               0.1381          6.3188
```

Distance equation (train, full):

- gates: {'close*adv': -5.613, 'close*wd': -3.049, 'flee*wd': 0.732, 'close*hold*far': -3.417, 'close*hold*contact': -5.285}
- rule: {'walk_r*wd': -0.337, 'danger*wd': 0.915, 'close*adv': -5.613, 'close*hold*far': -3.417, 'close*hold*contact': -5.285}

Withdraw coefficients across refits:

```
                              close*wd  flee*wd  walk_r*wd  danger*wd
gates train (full)              -3.049    0.732        NaN        NaN
      val (refit)              -12.845    2.976        NaN        NaN
      train, 500 battles (a)    -3.028    0.633        NaN        NaN
      train, 500 battles (b)    -2.981    0.781        NaN        NaN
rule  train (full)                 NaN      NaN     -0.337      0.915
      val (refit)                  NaN      NaN     -0.393      0.954
      train, 500 battles (a)       NaN      NaN     -0.317      0.872
      train, 500 battles (b)       NaN      NaN     -0.361      0.957
```

## mixed (withdraw radius 8.00)

Start points withdrawing > 50% of the first 2 s: 3044 of 14520.

```
       distance_1s  distance_2s  distance_4s  distance_1s, withdrawing  distance_2s, withdrawing  distance_4s, withdrawing  health_2s  health_4s  margin_r2_2s  battle_health_nrmse  duration_err_s
gates       0.3017       0.4361       0.6076                    0.5952                    0.7944                    0.8436     0.0346     0.0641        0.3707               0.1337          4.7332
rule        0.3014       0.4318       0.5986                    0.5927                    0.7935                    0.8372     0.0346     0.0640        0.3743               0.1300          4.5432
```

Distance equation (train, full):

- gates: {'close*adv': -5.434, 'close*wd': -3.05, 'flee*wd': 0.445, 'close*hold*far': -3.081, 'close*hold*contact': -5.263}
- rule: {'walk_r*wd': -0.34, 'danger*wd': 0.708, 'close*adv': -5.434, 'close*hold*far': -3.081, 'close*hold*contact': -5.263}

Withdraw coefficients across refits:

```
                              close*wd   flee*wd                                                    fit failed walk_r*wd danger*wd
gates train (full)           -3.050165  0.444737                                                           NaN       NaN       NaN
      val (refit)            -3.048043  0.464464                                                           NaN       NaN       NaN
      train, 500 battles (a) -3.013823  0.465602                                                           NaN       NaN       NaN
      train, 500 battles (b)       NaN       NaN  Optimal parameters not found: Number of calls to function ha       NaN       NaN
rule  train (full)                 NaN       NaN                                                           NaN -0.339983  0.707985
      val (refit)                  NaN       NaN                                                           NaN -0.264602  0.670891
      train, 500 battles (a)       NaN       NaN                                                           NaN -0.411482  0.780837
      train, 500 battles (b)       NaN       NaN                                                           NaN  -0.34007  0.717887
```
