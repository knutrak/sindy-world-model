# What if withdraw were removed? (imitation with existing data, 20 Nov)

Script: `experiments/mixed/no_withdraw.py`. Battles that never withdraw: marines 45% (train 2686,
val 455), mixed 33% (train 2006, val 329). See the script's docstring for the caveat.

## Marines (exp06), A*-2k

```
                                           health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  n_battles  mean battle s
fitted on all train   scored on all val        0.028      0.053         0.409         0.087     0.922        0.492                0.129       0.651           3.051     1000.0         10.389
                      scored on no-wd val      0.027      0.049         0.420         0.071     0.953        0.196                0.098       0.725           1.401      455.0          7.674
fitted on no-wd train scored on no-wd val      0.026      0.046         0.520         0.274     0.908        0.200                0.088       0.760           1.552      455.0          7.674
```

Coefficients:

```
                              all train  no-wd train
health_blue: Rr                 -0.1770      -0.1475
health_red: Rb                  -0.1861      -0.1412
health_red: Rb*wd                0.1633          NaN
distance: close*adv             -6.0077      -6.2516
distance: close*wd              -3.1575          NaN
distance: flee*wd                0.2962          NaN
distance: close*hold*far        -3.1830      -3.5001
distance: close*hold*contact    -5.9147      -6.2619
alive_blue: 1                    0.6625       0.8179
alive_red: 1                     0.5394       0.7295
alive_red: focus                 0.2384       0.3557
health_blue: Ar                     NaN      -0.0227
health_red: Ab                      NaN      -0.0322
```

## Mixed (exp08, reach 6)

```
                                                           health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  n_battles  mean battle s
team A*-2k      fitted on all train   scored on all val        0.039      0.072         0.335         0.039     0.771        0.609                0.141       0.671           5.256     1000.0         15.949
typed + wd rule fitted on all train   scored on all val        0.034      0.065         0.370         0.101     0.787        0.608                0.139       0.663           4.975     1000.0         15.949
team A*-2k      fitted on all train   scored on no-wd val      0.041      0.072         0.435         0.305     0.811        0.232                0.115       0.760           2.517      329.0         11.017
typed + wd rule fitted on all train   scored on no-wd val      0.034      0.061         0.497         0.329     0.902        0.232                0.112       0.754           2.554      329.0         11.017
team A*-2k      fitted on no-wd train scored on no-wd val      0.041      0.068         0.471         0.385     0.838        0.247                0.111       0.766           2.413      329.0         11.017
typed + wd rule fitted on no-wd train scored on no-wd val      0.034      0.062         0.509         0.348     0.880        0.247                0.105       0.760           2.369      329.0         11.017
```

Team model coefficients:

```
                              all train  no-wd train
health_blue: Rr                 -0.2177      -0.2150
health_red: Rb                  -0.2195      -0.1736
health_red: Rb*wd                0.1855          NaN
distance: close*adv             -5.4331      -6.0648
distance: close*wd              -3.5966          NaN
distance: flee*wd                0.4691          NaN
distance: close*hold*far        -3.0806      -3.4983
distance: close*hold*contact    -5.2624      -5.7499
alive_blue: 1                    0.6391       0.6753
alive_red: 1                     0.5521       0.6951
alive_red: focus                 0.2557       0.3783
health_red: Ab                      NaN      -0.0315
```
