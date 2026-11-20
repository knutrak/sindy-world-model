# WP4: the pipeline in three scenarios on val (20 Nov)

Script: `experiments/generalisation/evaluate.py`. Fitted on each scenario's train set, scored on
its val set. Health in ten units of the scenario's first type.

## marines

```
             health_1s  health_2s  health_4s  health_8s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate
wd fitted       0.0166     0.0281     0.0533     0.0959        0.4051        0.0725    0.9236       0.4925               0.1301       0.648          3.0936          0.0
wd rule         0.0164     0.0279     0.0534     0.0971        0.3818        0.0294    0.9234       0.4925               0.1342       0.641          3.2198          0.0
persistence     0.0622     0.1232     0.2334     0.3412       -0.0214       -0.0278    2.9544       0.8949               0.5598       0.633         79.6112          0.0
Lanchester      0.0366     0.0640     0.1024     0.1339       -0.0478       -0.3569    1.4544       0.8949               0.1812       0.638          4.3058          0.0
```

Gates (G3a light: 10 refits; G5 vs every baseline):

```
             G1    G2    G4    G5 G6 horizon (s)   G3a
wd fitted  True  True  True  True              2  True
wd rule    True  True  True  True              2  True
```

- **wd fitted**: one-step R² marine (blue) 0.717, marine (red) 0.726; term counts {'health_blue_marine': 1, 'health_red_marine': 2, 'distance': 5}; sign failures none
- **wd rule**: one-step R² marine (blue) 0.717, marine (red) 0.724; term counts {'health_blue_marine': 1, 'health_red_marine': 1, 'distance': 5}; sign failures none

Coefficients against the game (wd fitted):

```
                             fitted   game  fitted / game
term                                                     
marine (red) → marine         0.177  0.213          0.830
marine (blue) → marine        0.186  0.213          0.873
distance close*adv            6.008  6.300          0.954
distance close*wd             3.157  3.150          1.002
distance close*hold*far       3.183  3.150          1.010
distance close*hold*contact   5.915  6.300          0.939
```

## marauders

```
             health_1s  health_2s  health_4s  health_8s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate
wd fitted       0.0119     0.0204     0.0386     0.0670        0.3450        0.0940    0.7265       0.6394               0.1377       0.624          6.3992          0.0
wd rule         0.0117     0.0203     0.0386     0.0675        0.3437        0.0797    0.7259       0.6394               0.1393       0.618          6.3122          0.0
persistence     0.0361     0.0717     0.1398     0.2538       -0.0027       -0.0052    1.7735       1.0848               0.5723       0.636         72.8888          0.0
Lanchester      0.0217     0.0371     0.0614     0.0907       -0.0151       -0.1749    1.0525       1.0848               0.1670       0.624          6.4848          0.0
```

Gates (G3a light: 10 refits; G5 vs every baseline):

```
             G1    G2    G4    G5 G6 horizon (s)   G3a
wd fitted  True  True  True  True              2  True
wd rule    True  True  True  True              2  True
```

- **wd fitted**: one-step R² marauder (blue) 0.338, marauder (red) 0.381; term counts {'health_blue_marauder': 1, 'health_red_marauder': 2, 'distance': 5}; sign failures none
- **wd rule**: one-step R² marauder (blue) 0.338, marauder (red) 0.379; term counts {'health_blue_marauder': 1, 'health_red_marauder': 1, 'distance': 5}; sign failures none

Coefficients against the game (wd fitted):

```
                             fitted   game  fitted / game
term                                                     
marauder (red) → marauder     0.125  0.135          0.922
marauder (blue) → marauder    0.126  0.135          0.930
distance close*adv            5.613  6.300          0.891
distance close*wd             3.049  3.150          0.968
distance close*hold*far       3.417  3.150          1.085
distance close*hold*contact   5.285  6.300          0.839
```

## mixed

```
                         health_1s  health_2s  health_4s  health_8s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate
wd fitted, closure          0.0217     0.0378     0.0707     0.1262        0.2624       -0.0159    0.7852       0.6074               0.1475       0.708          5.1016          0.0
wd fitted, proportional     0.0220     0.0381     0.0703     0.1232        0.2557        0.0129    0.7637       0.6076               0.1427       0.726          5.2011          0.0
wd rule, closure            0.0198     0.0344     0.0650     0.1186        0.3654        0.0923    0.7539       0.6073               0.1377       0.660          4.6486          0.0
wd rule, proportional       0.0200     0.0346     0.0641     0.1143        0.3707        0.1373    0.7389       0.6076               0.1337       0.667          4.7332          0.0
persistence                 0.0641     0.1267     0.2451     0.4359       -0.0330       -0.0504    1.9288       0.9692               0.5674       0.638         74.0506          0.0
Lanchester                  0.0392     0.0685     0.1136     0.1652       -0.0130       -0.1945    1.1307       0.9692               0.1789       0.672          6.4664          0.0
team-level A*-2k            0.0215     0.0386     0.0718     0.1222        0.3355        0.0386    0.7711       0.6088               0.1414       0.671          5.2564          0.0
```

Gates (G3a light: 10 refits; G5 vs every baseline):

```
                           G1    G2    G4    G5 G6 horizon (s)   G3a
wd fitted, closure       True  True  True  True              1  True
wd fitted, proportional  True  True  True  True           None  True
wd rule, closure         True  True  True  True              2  True
wd rule, proportional    True  True  True  True              2  True
```

- **wd fitted, closure**: one-step R² marine (blue) 0.554, marauder (blue) 0.425, marine (red) 0.594, marauder (red) 0.491; term counts {'health_blue_marine': 2, 'health_blue_marauder': 2, 'health_red_marine': 2, 'health_red_marauder': 2, 'distance': 5}; sign failures none
- **wd fitted, proportional**: one-step R² marine (blue) 0.554, marauder (blue) 0.425, marine (red) 0.594, marauder (red) 0.491; term counts {'health_blue_marine': 2, 'health_blue_marauder': 2, 'health_red_marine': 2, 'health_red_marauder': 2, 'distance': 5}; sign failures none
- **wd rule, closure**: one-step R² marine (blue) 0.554, marauder (blue) 0.425, marine (red) 0.621, marauder (red) 0.509; term counts {'health_blue_marine': 2, 'health_blue_marauder': 2, 'health_red_marine': 2, 'health_red_marauder': 2, 'distance': 5}; sign failures none
- **wd rule, proportional**: one-step R² marine (blue) 0.554, marauder (blue) 0.425, marine (red) 0.621, marauder (red) 0.509; term counts {'health_blue_marine': 2, 'health_blue_marauder': 2, 'health_red_marine': 2, 'health_red_marauder': 2, 'distance': 5}; sign failures none

Coefficients against the game (wd fitted, closure):

```
                             fitted   game  fitted / game
term                                                     
marine (red) → marine         0.178  0.213          0.836
marine (red) → marauder       0.060  0.064          0.937
marauder (red) → marine       0.165  0.198          0.836
marauder (red) → marauder     0.113  0.135          0.834
marine (blue) → marine        0.184  0.213          0.865
marine (blue) → marauder      0.060  0.064          0.932
marauder (blue) → marine      0.163  0.198          0.823
marauder (blue) → marauder    0.112  0.135          0.832
distance close*adv            5.434  6.300          0.862
distance close*wd             3.050  3.150          0.968
distance close*hold*far       3.081  3.150          0.978
distance close*hold*contact   5.263  6.300          0.835
```

## WP5: applying the decision rules

```
     scenario               metric     option   value  rank
0     marines            health_4s  wd fitted  0.0533   1.0
1     marines            health_4s    wd rule  0.0534   2.0
2     marines  battle_health_nrmse  wd fitted  0.1301   1.0
3     marines  battle_health_nrmse    wd rule  0.1342   2.0
4   marauders            health_4s  wd fitted  0.0386   2.0
5   marauders            health_4s    wd rule  0.0386   1.0
6   marauders  battle_health_nrmse  wd fitted  0.1377   1.0
7   marauders  battle_health_nrmse    wd rule  0.1393   2.0
8       mixed            health_4s  wd fitted  0.0703   2.0
9       mixed            health_4s    wd rule  0.0641   1.0
10      mixed  battle_health_nrmse  wd fitted  0.1427   2.0
11      mixed  battle_health_nrmse    wd rule  0.1337   1.0
```

Mean rank: {'wd fitted': 1.5, 'wd rule': 1.5} → **withdraw: wd rule**.
Mixed, with wd rule: closure health 4 s 0.0650 / nRMSE 0.1377; proportional 0.0641 / 0.1337 → **target choice: proportional**.
