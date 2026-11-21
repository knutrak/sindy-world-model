# Headroom test, Phase A: results (run 20-21 Nov 2026)

Plan: `00_headroom_plan.md`. Data: `data/processed/planning/headroom_<scenario>.csv`. Script:
`experiments/planning/analyse.py`. Fixed intents: 400 seeds; oracles: 200 seeds (paired comparisons on
those). Oracles re-plan every 1 s, commit 2 s, horizon 8 s, optimise the health margin (the survival
oracle optimises Blue units preserved). Health margin = Blue − Red health left, as shares of full health.

Notes on the run:

- **Duplicates removed.** A stopped run kept going beside its restart, so S3-S5 oracles (and the S4
  survival oracle) were played twice. Every duplicate pair had identical outcomes (the games are
  deterministic per seed); one copy of each was kept.
- **O3 (units preserved) is won by always withdrawing,** which never wins a battle. Read O3 together
  with O1.
- **The survival oracle cannot value a sustained withdraw:** each candidate is played for 2 s, then
  advance/nearest to the 8 s horizon, so withdrawing only ever looks like a 2 s delay. Its O3 check
  below says little about how much a survival-minded planner could gain.
- **Deviation from the plan, checked.** The plan gave each perturbed copy its own random seed; the code
  only perturbed positions and health, so copies shared the real game's stream for SMAClite's per-tick
  unit-order shuffle (its only in-battle randomness). Rerun with own streams (`reseed=True`, 100 paired
  seeds, `reseed_check.py`): win change S1 −0.07 [−0.15, +0.01], S3 +0.05 [+0.01, +0.10]; margin within
  ±0.013, intervals across 0. No consistent extra foresight; headroom over the best fixed intent stays
  +0.11 [+0.02, +0.19] (S1) and +0.47 [+0.38, +0.57] (S3) on win. The results below are kept as run.
- **Withdraw ignores the target rule,** so the oracles have five distinct candidates; withdraw is
  logged as withdraw/nearest.

## S1

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'oracle': 200, 'oracle-perturbed': 200, 'random': 400}

```
                          won  margin  blue_preserved  log_exchange
policy                                                             
fixed:advance/focus     0.670   0.072           0.259         0.191
fixed:advance/nearest   0.602   0.052           0.236         0.079
fixed:hold/focus        0.685   0.084           0.274         0.227
fixed:hold/nearest      0.638   0.065           0.249         0.125
fixed:withdraw/focus    0.000  -0.221           0.788        -1.382
fixed:withdraw/nearest  0.000  -0.221           0.788        -1.382
oracle                  0.940   0.260           0.522         0.743
oracle-perturbed        0.865   0.188           0.427         0.536
random                  0.598   0.048           0.253         0.134
```

Intent use by the oracles (share of steps):

```
                  share_advance/nearest  share_advance/focus  share_hold/nearest  share_hold/focus  share_withdraw/nearest  share_withdraw/focus
policy                                                                                                                                          
oracle                            0.459                0.350               0.137             0.125                   0.282                   NaN
oracle-perturbed                  0.211                0.255               0.227             0.229                   0.268                   NaN
```

## S2

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'oracle': 200, 'oracle-perturbed': 200, 'random': 400}

```
                          won  margin  blue_preserved  log_exchange
policy                                                             
fixed:advance/focus     0.602   0.049           0.221         0.159
fixed:advance/nearest   0.545   0.030           0.206         0.033
fixed:hold/focus        0.590   0.050           0.215         0.151
fixed:hold/nearest      0.565   0.032           0.210         0.042
fixed:withdraw/focus    0.000  -0.168           0.866        -0.988
fixed:withdraw/nearest  0.000  -0.168           0.866        -0.988
oracle                  0.965   0.292           0.515         0.744
oracle-perturbed        0.860   0.196           0.398         0.490
random                  0.598   0.053           0.243         0.162
```

Intent use by the oracles (share of steps):

```
                  share_advance/nearest  share_advance/focus  share_hold/nearest  share_hold/focus  share_withdraw/nearest  share_withdraw/focus
policy                                                                                                                                          
oracle                            0.512                0.306               0.137             0.083                   0.194                   NaN
oracle-perturbed                  0.208                0.236               0.201             0.237                   0.189                   NaN
```

## S3

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'oracle': 200, 'oracle-perturbed': 200, 'random': 400}

```
                          won  margin  blue_preserved  log_exchange
policy                                                             
fixed:advance/focus     0.515   0.003           0.246         0.659
fixed:advance/nearest   0.542   0.032           0.285         0.626
fixed:hold/focus        0.545   0.031           0.264         0.710
fixed:hold/nearest      0.535   0.032           0.275         0.629
fixed:withdraw/focus    0.000  -0.118           0.916        -0.456
fixed:withdraw/nearest  0.000  -0.118           0.916        -0.456
oracle                  0.980   0.398           0.792         2.087
oracle-perturbed        0.935   0.341           0.705         1.848
random                  0.668   0.104           0.384         0.938
```

Intent use by the oracles (share of steps):

```
                  share_advance/nearest  share_advance/focus  share_hold/nearest  share_hold/focus  share_withdraw/nearest  share_withdraw/focus
policy                                                                                                                                          
oracle                            0.456                0.263               0.134             0.065                   0.253                   NaN
oracle-perturbed                  0.177                0.204               0.188             0.204                   0.266                   NaN
```

## S4

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'oracle': 200, 'oracle-perturbed': 200, 'oracle-survival': 200, 'random': 400}

```
                          won  margin  blue_preserved  log_exchange
policy                                                             
fixed:advance/focus     0.080  -0.298           0.019        -0.449
fixed:advance/nearest   0.080  -0.270           0.023        -0.720
fixed:hold/focus        0.105  -0.280           0.022        -0.411
fixed:hold/nearest      0.095  -0.261           0.024        -0.688
fixed:withdraw/focus    0.000  -0.375           0.831        -1.151
fixed:withdraw/nearest  0.000  -0.375           0.831        -1.151
oracle                  0.650   0.047           0.275         0.220
oracle-perturbed        0.465  -0.049           0.165        -0.044
oracle-survival         0.575   0.002           0.317         0.204
random                  0.115  -0.263           0.038        -0.548
```

Intent use by the oracles (share of steps):

```
                  share_advance/nearest  share_advance/focus  share_hold/nearest  share_hold/focus  share_withdraw/nearest  share_withdraw/focus
policy                                                                                                                                          
oracle                            0.502                0.248               0.127             0.072                   0.237                   NaN
oracle-perturbed                  0.214                0.194               0.213             0.194                   0.244                   NaN
oracle-survival                   0.156                0.140               0.212             0.199                   0.331                   NaN
```

## S5

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'oracle': 200, 'oracle-perturbed': 200, 'random': 400}

```
                          won  margin  blue_preserved  log_exchange
policy                                                             
fixed:advance/focus     0.552   0.040           0.279         0.131
fixed:advance/nearest   0.530   0.043           0.301         0.030
fixed:hold/focus        0.552   0.046           0.282         0.149
fixed:hold/nearest      0.532   0.045           0.301         0.032
fixed:withdraw/focus    0.000  -0.161           0.855        -1.051
fixed:withdraw/nearest  0.000  -0.161           0.855        -1.051
oracle                  0.790   0.256           0.497         0.731
oracle-perturbed        0.705   0.186           0.413         0.486
random                  0.558   0.060           0.317         0.190
```

Intent use by the oracles (share of steps):

```
                  share_advance/nearest  share_advance/focus  share_hold/nearest  share_hold/focus  share_withdraw/nearest  share_withdraw/focus
policy                                                                                                                                          
oracle                            0.492                0.298               0.138             0.084                   0.206                   NaN
oracle-perturbed                  0.221                0.229               0.211             0.224                   0.206                   NaN
```

## Headroom per scenario and objective

```
                                       best fixed  best fixed value  oracle  oracle − best fixed oracle − best fixed, 95%  oracle-perturbed  oracle-perturbed − best fixed oracle-perturbed − best fixed, 95%  foresight share  oracle-survival  oracle-survival − best fixed oracle-survival − best fixed, 95%
scenario objective                                                                                                                                                                                                                                                                                             
S1       O1 win                        hold/focus             0.685   0.940                0.240         [+0.175, +0.300]             0.865                          0.165                   [+0.095, +0.230]            0.312              NaN                           NaN                               NaN
         O2 health margin              hold/focus             0.084   0.260                0.173         [+0.149, +0.197]             0.188                          0.100                   [+0.075, +0.124]            0.419              NaN                           NaN                               NaN
         O3 units preserved        withdraw/focus             0.788   0.522               -0.253         [-0.282, -0.225]             0.427                         -0.348                   [-0.381, -0.312]           -0.373              NaN                           NaN                               NaN
         O4 exchange ratio (log)       hold/focus             0.227   0.743                0.515         [+0.442, +0.587]             0.536                          0.308                   [+0.231, +0.383]            0.402              NaN                           NaN                               NaN
S2       O1 win                     advance/focus             0.602   0.965                0.320         [+0.255, +0.390]             0.860                          0.215                   [+0.155, +0.280]            0.328              NaN                           NaN                               NaN
         O2 health margin              hold/focus             0.050   0.292                0.240         [+0.210, +0.270]             0.196                          0.144                   [+0.118, +0.171]            0.399              NaN                           NaN                               NaN
         O3 units preserved        withdraw/focus             0.866   0.515               -0.353         [-0.377, -0.328]             0.398                         -0.471                   [-0.503, -0.439]           -0.333              NaN                           NaN                               NaN
         O4 exchange ratio (log)    advance/focus             0.159   0.744                0.565         [+0.490, +0.644]             0.490                          0.311                   [+0.244, +0.377]            0.450              NaN                           NaN                               NaN
S3       O1 win                        hold/focus             0.545   0.980                0.420         [+0.355, +0.495]             0.935                          0.375                   [+0.310, +0.445]            0.107              NaN                           NaN                               NaN
         O2 health margin         advance/nearest             0.032   0.398                0.377         [+0.342, +0.409]             0.341                          0.320                   [+0.285, +0.351]            0.152              NaN                           NaN                               NaN
         O3 units preserved        withdraw/focus             0.916   0.792               -0.127         [-0.155, -0.101]             0.705                         -0.215                   [-0.252, -0.180]           -0.688              NaN                           NaN                               NaN
         O4 exchange ratio (log)       hold/focus             0.710   2.087                1.364         [+1.253, +1.478]             1.848                          1.124                   [+1.014, +1.244]            0.176              NaN                           NaN                               NaN
S4       O1 win                        hold/focus             0.105   0.650                0.550         [+0.475, +0.620]             0.465                          0.365                   [+0.295, +0.435]            0.336            0.575                         0.475                  [+0.405, +0.555]
         O2 health margin            hold/nearest            -0.261   0.047                0.305         [+0.269, +0.340]            -0.049                          0.208                   [+0.175, +0.242]            0.316            0.002                         0.260                  [+0.221, +0.300]
         O3 units preserved        withdraw/focus             0.831   0.275               -0.566         [-0.597, -0.536]             0.165                         -0.677                   [-0.707, -0.647]           -0.194            0.317                        -0.524                  [-0.559, -0.491]
         O4 exchange ratio (log)       hold/focus            -0.411   0.220                0.643         [+0.562, +0.730]            -0.044                          0.379                   [+0.309, +0.459]            0.410            0.204                         0.626                  [+0.500, +0.750]
S5       O1 win                     advance/focus             0.552   0.790                0.230         [+0.175, +0.290]             0.705                          0.145                   [+0.090, +0.195]            0.370              NaN                           NaN                               NaN
         O2 health margin              hold/focus             0.046   0.256                0.204         [+0.177, +0.232]             0.186                          0.135                   [+0.112, +0.159]            0.339              NaN                           NaN                               NaN
         O3 units preserved        withdraw/focus             0.855   0.497               -0.366         [-0.407, -0.324]             0.413                         -0.450                   [-0.491, -0.410]           -0.229              NaN                           NaN                               NaN
         O4 exchange ratio (log)       hold/focus             0.149   0.731                0.574         [+0.494, +0.651]             0.486                          0.328                   [+0.254, +0.402]            0.429              NaN                           NaN                               NaN
```

## Go / no-go for Phase B (rule: perturbed oracle ≥ best fixed + 5 points, 95% interval above 0)

- **S1: GO** — oracle-perturbed on won: +0.165 (lower +0.095)
- **S2: GO** — oracle-perturbed on won: +0.215 (lower +0.155)
- **S3: GO** — oracle-perturbed on won: +0.375 (lower +0.310)
- **S4: GO** — oracle-perturbed on won: +0.365 (lower +0.295); oracle-survival on blue_preserved: -0.524 (lower -0.559)
- **S5: GO** — oracle-perturbed on won: +0.145 (lower +0.090)
