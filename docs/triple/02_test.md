# Three unit types: held-out test (run once)

Plan: `00_plan.md`. Script: `experiments/triple/triple.py`. Fitted on exp12 train (1500 battles).

## Equations

```
H_B^marine' = -0.1820·R[marine]*s[marine] -0.1537·R[marauder]*s[marine] -0.1667·R[stalker]*s[marine]
H_B^marauder' = -0.0649·R[marine]*s[marauder] -0.1136·R[marauder]*s[marauder] -0.0899·R[stalker]*s[marauder]
H_B^stalker' = -0.0486·R[marine]*s[stalker] -0.0909·R[marauder]*s[stalker] -0.0701·R[stalker]*s[stalker]
H_R^marine' = -0.1923·R[marine]*s[marine] -0.1629·R[marauder]*s[marine] -0.1661·R[stalker]*s[marine]
H_R^marauder' = -0.0616·R[marine]*s[marauder] -0.1158·R[marauder]*s[marauder] -0.0883·R[stalker]*s[marauder]
H_R^stalker' = -0.0522·R[marine]*s[stalker] -0.0913·R[marauder]*s[stalker] -0.0717·R[stalker]*s[stalker]
(ln A_B^marine)' = (0.477) · (ln H_B^marine)'
(ln A_B^marauder)' = (0.425) · (ln H_B^marauder)'
(ln A_B^stalker)' = (0.449) · (ln H_B^stalker)'
(ln A_R^marine)' = (0.407 +0.115·focus) · (ln H_R^marine)'
(ln A_R^marauder)' = (0.378 +0.117·focus) · (ln H_R^marauder)'
(ln A_R^stalker)' = (0.349 +0.152·focus) · (ln H_R^stalker)'
d' = -0.247·walk_r*wd +0.775·danger*wd -5.535·close*adv -3.189·close*hold*far -5.151·close*hold*contact
```

## Prediction (health in ten marines' hp, mean per side)

```
                  health_1s  health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_2s  alive_4s  distance_2s  battle_health_nrmse  winner_acc  duration_err_s  blowup_rate  n_battles
pipeline             0.0261     0.0426     0.0778        0.4543        0.3514    0.4729    0.6677       0.4241               0.1328       0.668          5.9330          0.0      500.0
persistence          0.0753     0.1485     0.2908       -0.0457       -0.0652    0.7589    1.5070       0.6255               0.5652       0.648         69.6098          0.0      500.0
Lanchester           0.0471     0.0806     0.1361        0.0110       -0.1085    0.6829    0.9783       0.6255               0.1704       0.678          7.3982          0.0      500.0
team-level A*-2k     0.0271     0.0463     0.0863        0.4183        0.2293    0.4848    0.6727       0.4373               0.1459       0.672          6.7190          0.0      500.0
```

Paired 95% (pipeline − baseline), health error: persistence 2 s [-0.1114, -0.1010]; persistence 4 s [-0.2253, -0.2020]; Lanchester 2 s [-0.0410, -0.0357]; Lanchester 4 s [-0.0643, -0.0528]; team-level A*-2k 2 s [-0.0044, -0.0031]; team-level A*-2k 4 s [-0.0106, -0.0066]

Reliable horizon (G6): 4 s. One-step R²: marine (blue) 0.40, marauder (blue) 0.38, stalker (blue) 0.38, marine (red) 0.46, marauder (red) 0.47, stalker (red) 0.48

## Firepower against the game (refitted on test)

```
                            fitted   game  fitted / game
term                                                    
marine (red) → marine        0.178  0.213          0.834
marine (red) → marauder      0.062  0.064          0.967
marine (red) → stalker       0.050  0.055          0.911
marauder (red) → marine      0.158  0.198          0.800
marauder (red) → marauder    0.111  0.135          0.825
marauder (red) → stalker     0.091  0.108          0.838
stalker (red) → marine       0.168  0.210          0.798
stalker (red) → marauder     0.089  0.099          0.904
stalker (red) → stalker      0.070  0.079          0.885
marine (blue) → marine       0.189  0.213          0.887
marine (blue) → marauder     0.063  0.064          0.990
marine (blue) → stalker      0.052  0.055          0.957
marauder (blue) → marine     0.163  0.198          0.824
marauder (blue) → marauder   0.116  0.135          0.861
marauder (blue) → stalker    0.093  0.108          0.860
stalker (blue) → marine      0.168  0.210          0.797
stalker (blue) → marauder    0.087  0.099          0.884
stalker (blue) → stalker     0.072  0.079          0.904
```

## Against the frozen mixed model (marines + marauders)

```
                            mixed  triple  ratio
term                                            
marine (red) → marine       0.178   0.178  0.997
marine (red) → marauder     0.060   0.062  1.032
marauder (red) → marine     0.165   0.158  0.957
marauder (red) → marauder   0.113   0.111  0.989
marine (blue) → marine      0.189   0.189  1.000
marine (blue) → marauder    0.060   0.063  1.058
marauder (blue) → marine    0.172   0.163  0.947
marauder (blue) → marauder  0.117   0.116  0.995
```

Reach closure R²: blue marine 0.77, blue marauder 0.81, blue stalker 0.84, red marine 0.57, red marauder 0.73, red stalker 0.72

Terms: {'health_blue_marine': 3, 'health_blue_marauder': 3, 'health_blue_stalker': 3, 'health_red_marine': 3, 'health_red_marauder': 3, 'health_red_stalker': 3, 'distance': 5}; terms without a rule or with a wrong sign: none

## Expectations

```
                                                                      met
T1 structure (3 own firepower terms each, no others; distance <= 6)  True
T2 physics (18 terms 0.75-1.0 x game)                                True
T3 beats persistence and Lanchester at 2 and 4 s                     True
T4 beats team-level A*-2k at 2 s                                     True
T5 reliable horizon >= 2 s                                           True
T6 health 2 s <= 1.2 x val                                           True
T7 shared terms within 10% of frozen mixed                           True
```
