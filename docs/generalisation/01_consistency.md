# WP2: the unified pipeline on marines against A*-2k (20 Nov)

Script: `experiments/generalisation/consistency.py`. Both fitted on exp06 train (6000 battles),
scored on exp06 val. Same start points as the harness: True.

## Coefficients

```
                                          A*-2k  pipeline, wd fitted  pipeline, wd rule
k_B (Blue's loss per Red unit in reach) -0.1770              -0.1770            -0.1770
k_R (Red's loss per Blue unit in reach) -0.1861              -0.1861            -0.1861
k_wd (Red's loss, Blue withdrawing)      0.1633               0.1633                NaN
p_B                                      0.6625               0.6625             0.6625
p_R                                      0.5394               0.5623             0.5623
p_focus (Red)                            0.2384               0.2597             0.2600
distance: close*adv                     -6.0077              -6.0077            -6.0077
distance: close*wd                      -3.1575              -3.1575            -3.1575
distance: flee*wd                        0.2962               0.2962             0.2962
distance: close*hold*far                -3.1830              -3.1831            -3.1831
distance: close*hold*contact            -5.9147              -5.9147            -5.9147
```

Other health terms kept by the pipeline: {'pipeline, wd fitted': {}, 'pipeline, wd rule': {}}

## Val metrics (team view, health in ten marines' hp)

```
                     health_1s  health_2s  health_4s  health_8s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate
A*-2k                   0.0166     0.0281     0.0532     0.0960        0.4092        0.0866    0.9220       0.4924               0.1286       0.651          3.1111          0.0
pipeline, wd fitted     0.0166     0.0281     0.0533     0.0959        0.4051        0.0725    0.9236       0.4925               0.1301       0.648          3.0936          0.0
pipeline, wd rule       0.0164     0.0279     0.0534     0.0971        0.3818        0.0294    0.9234       0.4925               0.1342       0.641          3.2198          0.0
```

## Differences, relative to A*-2k

```
                                         pipeline, wd fitted  pipeline, wd rule
k_B (Blue's loss per Red unit in reach)              -0.0000            -0.0000
k_R (Red's loss per Blue unit in reach)               0.0000             0.0000
k_wd (Red's loss, Blue withdrawing)                   0.0000                NaN
p_B                                                   0.0000             0.0000
p_R                                                   0.0425             0.0423
p_focus (Red)                                         0.0893             0.0905
distance: close*adv                                   0.0000             0.0000
distance: close*wd                                   -0.0000            -0.0000
distance: flee*wd                                     0.0001             0.0001
distance: close*hold*far                              0.0000             0.0000
distance: close*hold*contact                         -0.0000            -0.0000
```

```
                     health_1s  health_2s  health_4s  health_8s  margin_r2_2s  margin_r2_4s  alive_4s  distance_4s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate
pipeline, wd fitted     0.0000     0.0000     0.0001    -0.0001       -0.0040       -0.0141    0.0016       0.0001               0.0015      -0.003         -0.0175          0.0
pipeline, wd rule      -0.0002    -0.0002     0.0002     0.0011       -0.0274       -0.0572    0.0015       0.0001               0.0056      -0.010          0.1086          0.0
```
