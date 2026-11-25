# Three unit types: val

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
pipeline             0.0262     0.0431     0.0799        0.4203        0.2751    0.4819    0.7026       0.4475               0.1339       0.670          5.8240          0.0      500.0
persistence          0.0745     0.1474     0.2881       -0.0296       -0.0434    0.7668    1.5289       0.6624               0.5699       0.652         69.4748          0.0      500.0
Lanchester           0.0465     0.0791     0.1320        0.0036       -0.1114    0.6787    0.9624       0.6624               0.1720       0.682          7.4698          0.0      500.0
team-level A*-2k     0.0271     0.0458     0.0846        0.3911        0.1856    0.4885    0.6860       0.4612               0.1450       0.658          6.5560          0.0      500.0
```

Paired 95% (pipeline − baseline), health error: persistence 2 s [-0.1089, -0.0999]; persistence 4 s [-0.2187, -0.1988]; Lanchester 2 s [-0.0381, -0.0339]; Lanchester 4 s [-0.0568, -0.0476]; team-level A*-2k 2 s [-0.0032, -0.0022]; team-level A*-2k 4 s [-0.0062, -0.0033]

Reliable horizon (G6): 3 s. One-step R²: marine (blue) 0.40, marauder (blue) 0.38, stalker (blue) 0.38, marine (red) 0.46, marauder (red) 0.47, stalker (red) 0.48

## Firepower against the game (fitted on train)

```
                            fitted   game  fitted / game
term                                                    
marine (red) → marine        0.182  0.213          0.853
marine (red) → marauder      0.065  0.064          1.014
marine (red) → stalker       0.049  0.055          0.890
marauder (red) → marine      0.154  0.198          0.778
marauder (red) → marauder    0.114  0.135          0.841
marauder (red) → stalker     0.091  0.108          0.840
stalker (red) → marine       0.167  0.210          0.793
stalker (red) → marauder     0.090  0.099          0.908
stalker (red) → stalker      0.070  0.079          0.881
marine (blue) → marine       0.192  0.213          0.901
marine (blue) → marauder     0.062  0.064          0.962
marine (blue) → stalker      0.052  0.055          0.957
marauder (blue) → marine     0.163  0.198          0.825
marauder (blue) → marauder   0.116  0.135          0.857
marauder (blue) → stalker    0.091  0.108          0.844
stalker (blue) → marine      0.166  0.210          0.790
stalker (blue) → marauder    0.088  0.099          0.892
stalker (blue) → stalker     0.072  0.079          0.902
```

## Against the frozen mixed model (marines + marauders)

```
                            mixed  triple  ratio
term                                            
marine (red) → marine       0.178   0.182  1.020
marine (red) → marauder     0.060   0.065  1.083
marauder (red) → marine     0.165   0.154  0.931
marauder (red) → marauder   0.113   0.114  1.008
marine (blue) → marine      0.189   0.192  1.016
marine (blue) → marauder    0.060   0.062  1.028
marauder (blue) → marine    0.172   0.163  0.948
marauder (blue) → marauder  0.117   0.116  0.990
```

Reach closure R²: blue marine 0.77, blue marauder 0.81, blue stalker 0.84, red marine 0.57, red marauder 0.73, red stalker 0.72

Terms: {'health_blue_marine': 3, 'health_blue_marauder': 3, 'health_blue_stalker': 3, 'health_red_marine': 3, 'health_red_marauder': 3, 'health_red_stalker': 3, 'distance': 5}; terms without a rule or with a wrong sign: none

## Expectations

```
                                                                       met
T1 structure (3 own firepower terms each, no others; distance <= 6)   True
T2 physics (18 terms 0.75-1.0 x game)                                False
T3 beats persistence and Lanchester at 2 and 4 s                      True
T4 beats team-level A*-2k at 2 s                                      True
T5 reliable horizon >= 2 s                                            True
T6 health 2 s <= 1.2 x val                                            None
T7 shared terms within 10% of frozen mixed                            True
```
