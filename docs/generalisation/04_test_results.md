# Generalisation test: held-out results (run once)

Pre-registration: `03_preregistration.md`. Script: `experiments/generalisation/heldout_test.py`.

## marines (`exp06_intents/test4`)

```
             health_1s  health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_2s  alive_4s  distance_2s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate  n_battles
pipeline        0.0161     0.0275     0.0507        0.3993        0.0482    0.5618    0.9020       0.3625               0.1293       0.676          3.0330          0.0     1000.0
persistence     0.0609     0.1204     0.2275       -0.0287       -0.0406    1.5019    2.8467       0.5897               0.5666       0.631         79.3609          0.0     1000.0
Lanchester      0.0360     0.0621     0.0976       -0.0537       -0.3624    0.9748    1.4159       0.5897               0.1828       0.656          4.5740          0.0     1000.0
```

Paired 95% (pipeline − baseline), health error:

```
                            interval
persistence, 2 s  [-0.0970, -0.0889]
persistence, 4 s  [-0.1860, -0.1675]
Lanchester, 2 s   [-0.0362, -0.0332]
Lanchester, 4 s   [-0.0498, -0.0442]
```

Reliable horizon (G6): 2 s.

Refit on this held-out set: against the game:

```
                             fitted   game  fitted / game
term                                                     
marine (red) → marine         0.175  0.213          0.820
marine (blue) → marine        0.184  0.213          0.861
distance close*adv            5.985  6.300          0.950
distance close*wd            -0.000  3.150         -0.000
distance close*hold*far       3.222  3.150          1.023
distance close*hold*contact   5.855  6.300          0.929
```

Refit term counts {'health_blue_marine': 1, 'health_red_marine': 1, 'distance': 5}; sign failures none.

## marauders (`exp09_marauders/test`)

```
             health_1s  health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_2s  alive_4s  distance_2s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate  n_battles
pipeline        0.0116     0.0199     0.0372        0.3740        0.2258    0.4919    0.7213       0.4378               0.1335       0.618          5.3912          0.0      500.0
persistence     0.0374     0.0744     0.1456       -0.0033       -0.0058    0.9360    1.8567       0.7076               0.5827       0.604         72.9725          0.0      500.0
Lanchester      0.0217     0.0374     0.0627       -0.0262       -0.1999    0.7338    1.0581       0.7076               0.1677       0.604          6.0440          0.0      500.0
```

Paired 95% (pipeline − baseline), health error:

```
                            interval
persistence, 2 s  [-0.0572, -0.0519]
persistence, 4 s  [-0.1143, -0.1029]
Lanchester, 2 s   [-0.0188, -0.0163]
Lanchester, 4 s   [-0.0280, -0.0231]
```

Reliable horizon (G6): 3 s.

Refit on this held-out set: against the game:

```
                             fitted   game  fitted / game
term                                                     
marauder (red) → marauder     0.124  0.135          0.920
marauder (blue) → marauder    0.125  0.135          0.925
distance close*adv            5.762  6.300          0.915
distance close*wd            -0.000  3.150         -0.000
distance close*hold*far       3.276  3.150          1.040
distance close*hold*contact   5.126  6.300          0.814
```

Refit term counts {'health_blue_marauder': 1, 'health_red_marauder': 1, 'distance': 5}; sign failures none.

## mixed (`exp08_mixed/test3`)

```
                  health_1s  health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_2s  alive_4s  distance_2s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate  n_battles
pipeline             0.0200     0.0343     0.0646        0.4219        0.1935    0.5010    0.7420       0.4204               0.1314       0.659          4.7780          0.0     1000.0
persistence          0.0651     0.1289     0.2501       -0.0235       -0.0362    1.0033    1.9714       0.6316               0.5631       0.648         74.0923          0.0     1000.0
Lanchester           0.0400     0.0703     0.1176       -0.0175       -0.1997    0.7954    1.1794       0.6316               0.1813       0.666          6.5738          0.0     1000.0
team-level A*-2k     0.0215     0.0383     0.0719        0.3800        0.0802    0.5262    0.7811       0.4345               0.1446       0.645          5.3599          0.0     1000.0
```

Paired 95% (pipeline − baseline), health error:

```
                                 interval
persistence, 2 s       [-0.0981, -0.0913]
persistence, 4 s       [-0.1931, -0.1780]
Lanchester, 2 s        [-0.0376, -0.0344]
Lanchester, 4 s        [-0.0562, -0.0495]
team-level A*-2k, 2 s  [-0.0044, -0.0037]
team-level A*-2k, 4 s  [-0.0082, -0.0064]
```

Reliable horizon (G6): 3 s.

Refit on this held-out set: against the game:

```
                             fitted   game  fitted / game
term                                                     
marine (red) → marine         0.177  0.213          0.828
marine (red) → marauder       0.059  0.064          0.923
marauder (red) → marine       0.165  0.198          0.836
marauder (red) → marauder     0.112  0.135          0.828
marine (blue) → marine        0.188  0.213          0.879
marine (blue) → marauder      0.060  0.064          0.937
marauder (blue) → marine      0.170  0.198          0.859
marauder (blue) → marauder    0.116  0.135          0.857
distance close*adv            5.398  6.300          0.857
distance close*wd            -0.000  3.150         -0.000
distance close*hold*far       2.926  3.150          0.929
distance close*hold*contact   5.256  6.300          0.834
```

Refit term counts {'health_blue_marine': 2, 'health_blue_marauder': 2, 'health_red_marine': 2, 'health_red_marauder': 2, 'distance': 5}; sign failures none.

## uneven (`exp10_uneven/test`)

```
                  health_1s  health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_2s  alive_4s  distance_2s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate  n_battles
pipeline             0.0217     0.0378     0.0700        0.3931        0.2025    0.5107    0.7678       0.4493               0.1224       0.895          4.1448          0.0     1000.0
persistence          0.0704     0.1395     0.2696       -0.0170       -0.0298    1.0081    1.9588       0.6291               0.5161       0.926         75.6491          0.0     1000.0
Lanchester           0.0441     0.0784     0.1380       -1.1937       -2.3626    0.8737    1.3489       0.6291               0.2040       0.926          5.8475          0.0     1000.0
team-level A*-2k     0.0231     0.0412     0.0761        0.3694        0.1478    0.5330    0.7875       0.4549               0.1335       0.878          4.7150          0.0     1000.0
```

Paired 95% (pipeline − baseline), health error:

```
                                 interval
persistence, 2 s       [-0.1053, -0.0975]
persistence, 4 s       [-0.2077, -0.1903]
Lanchester, 2 s        [-0.0423, -0.0388]
Lanchester, 4 s        [-0.0718, -0.0641]
team-level A*-2k, 2 s  [-0.0039, -0.0030]
team-level A*-2k, 4 s  [-0.0072, -0.0050]
```

Reliable horizon (G6): 3 s.

Refit on this held-out set: against the game:

```
                             fitted   game  fitted / game
term                                                     
marine (red) → marine         0.164  0.213          0.770
marine (red) → marauder       0.063  0.064          0.990
marauder (red) → marine       0.161  0.198          0.813
marauder (red) → marauder     0.107  0.135          0.793
marine (blue) → marine        0.183  0.213          0.860
marine (blue) → marauder      0.065  0.064          1.020
marauder (blue) → marine      0.168  0.198          0.851
marauder (blue) → marauder    0.110  0.135          0.818
distance close*adv            5.604  6.300          0.889
distance close*wd            -0.000  3.150         -0.000
distance close*hold*far       3.062  3.150          0.972
distance close*hold*contact   5.511  6.300          0.875
```

Refit term counts {'health_blue_marine': 2, 'health_blue_marauder': 2, 'health_red_marine': 2, 'health_red_marauder': 2, 'distance': 5}; sign failures none.

## Expectations

```
                                                              met
scenario  expectation                                            
marines   E1 structure                                       True
          E2 physics                                         True
          E3 beats persistence and Lanchester at 2 and 4 s   True
          E5 reliable horizon >= 2 s                         True
          E6 health 2 s <= 1.2 x val                         True
marauders E1 structure                                       True
          E2 physics                                         True
          E3 beats persistence and Lanchester at 2 and 4 s   True
          E5 reliable horizon >= 2 s                         True
          E6 health 2 s <= 1.2 x val                         True
mixed     E1 structure                                       True
          E2 physics                                         True
          E3 beats persistence and Lanchester at 2 and 4 s   True
          E4 per-type beats team-level at 2 s                True
          E5 reliable horizon >= 2 s                         True
          E6 health 2 s <= 1.2 x val                         True
uneven    E1 structure                                       True
          E2 physics                                        False
          E3 beats persistence and Lanchester at 2 and 4 s   True
          E4 per-type beats team-level at 2 s                True
```

Met: 19 of 20.
