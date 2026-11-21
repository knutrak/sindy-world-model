# Phase B: the frozen model as a planner (results)

Plan: `07_phase_b_plan.md` (pre-registered, approved 21 Nov). Data: `data/processed/planning/phase_b_<S>.csv`.
Script: `experiments/planning/analyse_phase_b.py`. Planners re-plan every 1 s, commit 2 s; the primary has
a 3 s horizon. Seeds 14 x00 000 +; planners, fixed intents and random 400 per scenario, exact oracle 200.
Main scenarios S1, S2, S5; S3 and S4 are outside the mixed model's tested range.

## Primary result (O1 won, model:h3 − Phase A best fixed, 400 paired seeds)

```
           planner baseline (Phase A best fixed)  planner won  baseline won  difference               95% read-out    n
scenario                                                                                                               
S1        model:h3                    hold/focus        0.642         0.680      -0.038  [-0.070, -0.005]    loses  400
S2        model:h3                 advance/focus        0.648         0.625       0.022  [+0.003, +0.043]    beats  400
S3        model:h3                    hold/focus        0.765         0.598       0.168  [+0.115, +0.217]    beats  400
S4        model:h3                    hold/focus        0.085         0.098      -0.012  [-0.035, +0.010]     ties  400
S5        model:h3                 advance/focus        0.538         0.538       0.000  [-0.007, +0.007]     ties  400
```

## Share of the headroom captured (O1, on the exact oracle's 200 paired seeds)

```
          oracle won  baseline won  planner won  share captured               95%    n
scenario                                                                              
S1             0.925         0.705        0.650          -0.250  [-0.568, -0.021]  200
S2             0.970         0.635        0.660           0.075  [+0.000, +0.151]  200
S3             0.970         0.590        0.755           0.434  [+0.282, +0.575]  200
S4             0.615         0.095        0.080          -0.029  [-0.089, +0.027]  200
S5             0.825         0.565        0.565           0.000  [-0.061, +0.055]  200
```

## Secondary planners (O1, against the same baseline)

```
                            baseline (Phase A best fixed)  planner won  baseline won  difference               95% read-out    n
scenario planner                                                                                                                
S1       model:h2                              hold/focus        0.645         0.680      -0.035  [-0.068, -0.003]    loses  400
         model:h4                              hold/focus        0.645         0.680      -0.035  [-0.068, -0.003]    loses  400
         model-focuscost:h3                    hold/focus        0.678         0.680      -0.002  [-0.043, +0.037]     ties  400
S2       model:h2                           advance/focus        0.638         0.625       0.012  [+0.000, +0.025]     ties  400
         model:h4                           advance/focus        0.645         0.625       0.020  [-0.003, +0.045]     ties  400
         model-focuscost:h3                 advance/focus        0.612         0.625      -0.012  [-0.055, +0.035]     ties  400
S3       model:h2                              hold/focus        0.755         0.598       0.158  [+0.117, +0.200]    beats  400
         model:h4                              hold/focus        0.690         0.598       0.092  [+0.037, +0.145]    beats  400
         model-focuscost:h3                    hold/focus        0.740         0.598       0.142  [+0.092, +0.198]    beats  400
S4       model:h2                              hold/focus        0.080         0.098      -0.018  [-0.040, +0.003]     ties  400
         model:h4                              hold/focus        0.095         0.098      -0.002  [-0.028, +0.022]     ties  400
         model-focuscost:h3                    hold/focus        0.062         0.098      -0.035  [-0.062, -0.007]    loses  400
S5       model:h2                           advance/focus        0.540         0.538       0.002  [+0.000, +0.007]     ties  400
         model:h4                           advance/focus        0.540         0.538       0.002  [-0.005, +0.013]     ties  400
         model-focuscost:h3                 advance/focus        0.532         0.538      -0.005  [-0.028, +0.018]     ties  400
```

## S1

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'model-focuscost:h3': 400, 'model:h2': 400, 'model:h3': 400, 'model:h4': 400, 'oracle': 200, 'random': 400}

```
                        O1 won  O2 margin  O3 preserved  O4 log exchange
policy                                                                  
fixed:advance/focus      0.645      0.065         0.248            0.166
fixed:advance/nearest    0.585      0.043         0.229            0.048
fixed:hold/focus         0.680      0.089         0.275            0.223
fixed:hold/nearest       0.618      0.060         0.249            0.107
fixed:withdraw/focus     0.000     -0.218         0.791           -1.355
fixed:withdraw/nearest   0.000     -0.218         0.791           -1.355
model-focuscost:h3       0.678      0.072         0.273            0.170
model:h2                 0.645      0.067         0.253            0.169
model:h3                 0.642      0.066         0.254            0.166
model:h4                 0.645      0.065         0.252            0.165
oracle                   0.925      0.260         0.524            0.741
random                   0.598      0.050         0.247            0.126
```

Best fixed intent on Phase B's seeds (check): fixed:hold/focus 0.680.

Planner diagnostics (share of re-plans per pick; means per battle):

```
                    advance/nearest  advance/focus  hold/nearest  hold/focus  withdraw/nearest  replans  switches  gap_best_second  switches per re-plan
policy                                                                                                                                                  
model-focuscost:h3            0.315          0.246         0.291       0.128             0.020    9.672     3.672            0.001                 0.380
model:h2                      0.019          0.623         0.016       0.328             0.014    9.725     1.932            0.000                 0.199
model:h3                      0.019          0.609         0.018       0.328             0.027   10.418     2.278            0.001                 0.219
model:h4                      0.020          0.592         0.005       0.364             0.019    9.420     2.018            0.001                 0.214
```

Primary planner − baseline, secondary objectives:

```
                 difference               95%
objective                                    
O2 margin            -0.023  [-0.034, -0.012]
O3 preserved         -0.020  [-0.035, -0.006]
O4 log exchange      -0.057  [-0.085, -0.029]
```

## S2

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'model-focuscost:h3': 400, 'model:h2': 400, 'model:h3': 400, 'model:h4': 400, 'oracle': 200, 'random': 400}

```
                        O1 won  O2 margin  O3 preserved  O4 log exchange
policy                                                                  
fixed:advance/focus      0.625      0.058         0.238            0.199
fixed:advance/nearest    0.565      0.037         0.216            0.066
fixed:hold/focus         0.648      0.066         0.237            0.201
fixed:hold/nearest       0.598      0.049         0.234            0.103
fixed:withdraw/focus     0.000     -0.139         0.874           -0.931
fixed:withdraw/nearest   0.000     -0.139         0.874           -0.931
model-focuscost:h3       0.612      0.063         0.256            0.168
model:h2                 0.638      0.061         0.248            0.200
model:h3                 0.648      0.070         0.267            0.208
model:h4                 0.645      0.075         0.279            0.194
oracle                   0.970      0.300         0.517            0.767
random                   0.605      0.057         0.254            0.187
```

Best fixed intent on Phase B's seeds (check): fixed:hold/focus 0.647.

Planner diagnostics (share of re-plans per pick; means per battle):

```
                    advance/nearest  advance/focus  hold/nearest  hold/focus  withdraw/nearest  replans  switches  gap_best_second  switches per re-plan
policy                                                                                                                                                  
model-focuscost:h3            0.302          0.207         0.267       0.109             0.115   25.035     6.470            0.002                 0.258
model:h2                      0.011          0.579         0.033       0.328             0.049   17.008     3.558            0.001                 0.209
model:h3                      0.009          0.553         0.051       0.271             0.116   24.652     4.328            0.001                 0.176
model:h4                      0.009          0.541         0.062       0.268             0.119   26.138     5.355            0.002                 0.205
```

Primary planner − baseline, secondary objectives:

```
                 difference               95%
objective                                    
O2 margin             0.012  [+0.005, +0.019]
O3 preserved          0.029  [+0.020, +0.037]
O4 log exchange       0.009  [-0.006, +0.024]
```

## S3 (flagged: outside the model's tested range)

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'model-focuscost:h3': 400, 'model:h2': 400, 'model:h3': 400, 'model:h4': 400, 'oracle': 200, 'random': 400}

```
                        O1 won  O2 margin  O3 preserved  O4 log exchange
policy                                                                  
fixed:advance/focus      0.580      0.030         0.268            0.703
fixed:advance/nearest    0.532      0.034         0.283            0.631
fixed:hold/focus         0.598      0.043         0.277            0.736
fixed:hold/nearest       0.535      0.033         0.277            0.633
fixed:withdraw/focus     0.000     -0.112         0.920           -0.436
fixed:withdraw/nearest   0.000     -0.112         0.920           -0.436
model-focuscost:h3       0.740      0.211         0.678            1.226
model:h2                 0.755      0.177         0.483            1.115
model:h3                 0.765      0.222         0.660            1.262
model:h4                 0.690      0.206         0.717            1.226
oracle                   0.970      0.388         0.767            2.006
random                   0.678      0.103         0.383            0.935
```

Best fixed intent on Phase B's seeds (check): fixed:hold/focus 0.598.

Planner diagnostics (share of re-plans per pick; means per battle):

```
                    advance/nearest  advance/focus  hold/nearest  hold/focus  withdraw/nearest  replans  switches  gap_best_second  switches per re-plan
policy                                                                                                                                                  
model-focuscost:h3            0.068          0.164         0.393       0.178             0.197   32.878     7.095            0.005                 0.216
model:h2                      0.015          0.265         0.066       0.522             0.132   23.500     5.965            0.002                 0.254
model:h3                      0.004          0.244         0.004       0.562             0.185   30.055     5.608            0.005                 0.187
model:h4                      0.005          0.219         0.009       0.382             0.385   38.570     5.530            0.008                 0.143
```

Primary planner − baseline, secondary objectives:

```
                 difference               95%
objective                                    
O2 margin             0.179  [+0.154, +0.203]
O3 preserved          0.384  [+0.349, +0.418]
O4 log exchange       0.526  [+0.419, +0.632]
```

## S4 (flagged: outside the model's tested range)

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'model-focuscost:h3': 400, 'model:h2': 400, 'model:h3': 400, 'model:h4': 400, 'oracle': 200, 'random': 400}

```
                        O1 won  O2 margin  O3 preserved  O4 log exchange
policy                                                                  
fixed:advance/focus      0.078     -0.293         0.023           -0.444
fixed:advance/nearest    0.048     -0.281         0.011           -0.735
fixed:hold/focus         0.098     -0.282         0.026           -0.419
fixed:hold/nearest       0.052     -0.271         0.013           -0.691
fixed:withdraw/focus     0.000     -0.374         0.840           -1.124
fixed:withdraw/nearest   0.000     -0.374         0.840           -1.124
model-focuscost:h3       0.062     -0.253         0.065           -0.599
model:h2                 0.080     -0.286         0.043           -0.446
model:h3                 0.085     -0.262         0.068           -0.421
model:h4                 0.095     -0.249         0.102           -0.444
oracle                   0.615      0.024         0.244            0.127
random                   0.140     -0.250         0.050           -0.491
```

Best fixed intent on Phase B's seeds (check): fixed:hold/focus 0.098.

Planner diagnostics (share of re-plans per pick; means per battle):

```
                    advance/nearest  advance/focus  hold/nearest  hold/focus  withdraw/nearest  replans  switches  gap_best_second  switches per re-plan
policy                                                                                                                                                  
model-focuscost:h3            0.145          0.039         0.438       0.099             0.278   35.420     7.865            0.004                 0.222
model:h2                      0.014          0.259         0.035       0.524             0.168   23.462     4.100            0.001                 0.175
model:h3                      0.010          0.221         0.057       0.447             0.265   33.948     6.535            0.003                 0.193
model:h4                      0.009          0.206         0.114       0.426             0.245   38.590     8.225            0.004                 0.213
```

Primary planner − baseline, secondary objectives:

```
                 difference               95%
objective                                    
O2 margin             0.020  [+0.009, +0.031]
O3 preserved          0.042  [+0.032, +0.052]
O4 log exchange      -0.002  [-0.028, +0.024]
```

## S5

Battles: {'fixed:advance/focus': 400, 'fixed:advance/nearest': 400, 'fixed:hold/focus': 400, 'fixed:hold/nearest': 400, 'fixed:withdraw/focus': 400, 'fixed:withdraw/nearest': 400, 'model-focuscost:h3': 400, 'model:h2': 400, 'model:h3': 400, 'model:h4': 400, 'oracle': 200, 'random': 400}

```
                        O1 won  O2 margin  O3 preserved  O4 log exchange
policy                                                                  
fixed:advance/focus      0.538      0.029         0.273            0.107
fixed:advance/nearest    0.542      0.044         0.306            0.042
fixed:hold/focus         0.532      0.037         0.278            0.127
fixed:hold/nearest       0.545      0.048         0.309            0.061
fixed:withdraw/focus     0.000     -0.168         0.858           -1.043
fixed:withdraw/nearest   0.000     -0.168         0.858           -1.043
model-focuscost:h3       0.532      0.052         0.322            0.062
model:h2                 0.540      0.038         0.290            0.112
model:h3                 0.538      0.044         0.301            0.112
model:h4                 0.540      0.051         0.322            0.109
oracle                   0.825      0.245         0.473            0.655
random                   0.535      0.033         0.293            0.094
```

Best fixed intent on Phase B's seeds (check): fixed:hold/nearest 0.545.

Planner diagnostics (share of re-plans per pick; means per battle):

```
                    advance/nearest  advance/focus  hold/nearest  hold/focus  withdraw/nearest  replans  switches  gap_best_second  switches per re-plan
policy                                                                                                                                                  
model-focuscost:h3            0.283          0.193         0.327       0.059             0.138   22.462     5.515            0.002                 0.246
model:h2                      0.015          0.509         0.015       0.348             0.113   20.295     3.068            0.001                 0.151
model:h3                      0.010          0.498         0.033       0.313             0.146   24.580     4.378            0.002                 0.178
model:h4                      0.011          0.488         0.056       0.313             0.133   26.515     5.105            0.003                 0.193
```

Primary planner − baseline, secondary objectives:

```
                 difference               95%
objective                                    
O2 margin             0.015  [+0.009, +0.021]
O3 preserved          0.029  [+0.020, +0.037]
O4 log exchange       0.006  [-0.010, +0.021]
```

