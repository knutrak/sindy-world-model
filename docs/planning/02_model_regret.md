# Headroom follow-up, step 1: the frozen model's regret on the counterfactual sets

Script: `experiments/planning/model_regret.py` (see its docstring for the set-up). Score: change in the
team health margin (Blue − Red, in ten units of the first type) from the branch moment to h s (or the
battle's end). Regret = true best − true score of the pick, per state; 95% bootstrap over states. The
true branches play the intent for 4 s, then advance/nearest, so up to 4 s the truth is the intent alone.

## S1 (marines model, exp06_intents_typed/cf_*)

Largest state difference across a seed's branches at the branch moment: 0.00e+00.

### S1 at 1 s

522 states; mean spread (best − worst branch) 0.0698; all five tied in 1%.
Model − always advance/nearest, regret: +0.0118 [+0.0097, +0.0139].
Model's picks: advance_nearest 0.00, advance_focus 0.58, hold_nearest 0.00, hold_focus 0.41, withdraw_nearest 0.01.

```
                                             regret          regret 95% costly (> 0.05) picks the true best rank correlation
model                                       0.01697  [+0.0153, +0.0186]        0.063218            0.329502         0.238618
always advance/nearest                     0.005214  [+0.0043, +0.0062]         0.01341            0.689655              NaN
best fixed in hindsight (advance_nearest)  0.005214  [+0.0043, +0.0062]         0.01341            0.689655              NaN
random (expected)                          0.022606  [+0.0214, +0.0238]             NaN                 NaN              0.0
```

### S1 at 2 s

522 states; mean spread (best − worst branch) 0.1248; all five tied in 0%.
Model − always advance/nearest, regret: +0.0116 [+0.0083, +0.0150].
Model's picks: advance_nearest 0.01, advance_focus 0.56, hold_nearest 0.01, hold_focus 0.41, withdraw_nearest 0.02.

```
                                             regret          regret 95% costly (> 0.05) picks the true best rank correlation
model                                      0.022484  [+0.0201, +0.0250]        0.180077            0.396552         0.347683
always advance/nearest                       0.0109  [+0.0093, +0.0125]        0.055556            0.582375              NaN
best fixed in hindsight (advance_nearest)    0.0109  [+0.0093, +0.0125]        0.055556            0.582375              NaN
random (expected)                          0.038337  [+0.0363, +0.0404]             NaN                 NaN              0.0
```

### S1 at 4 s

522 states; mean spread (best − worst branch) 0.2091; all five tied in 0%.
Model − always advance/nearest, regret: +0.0041 [-0.0013, +0.0098].
Model's picks: advance_nearest 0.02, advance_focus 0.48, hold_nearest 0.03, hold_focus 0.43, withdraw_nearest 0.03.

```
                                             regret          regret 95% costly (> 0.05) picks the true best rank correlation
model                                       0.03078  [+0.0267, +0.0348]        0.252874            0.467433         0.387342
always advance/nearest                     0.026692  [+0.0235, +0.0300]        0.233716             0.45977              NaN
best fixed in hindsight (advance_nearest)  0.026692  [+0.0235, +0.0301]        0.233716             0.45977              NaN
random (expected)                          0.064014  [+0.0605, +0.0677]             NaN                 NaN              0.0
```

### S1 at 8 s

522 states; mean spread (best − worst branch) 0.1984; all five tied in 1%.
Model − always advance/nearest, regret: -0.0028 [-0.0105, +0.0047].
Model's picks: advance_nearest 0.02, advance_focus 0.41, hold_nearest 0.05, hold_focus 0.52, withdraw_nearest 0.00.

```
                                           regret          regret 95% costly (> 0.05) picks the true best rank correlation
model                                    0.057967  [+0.0507, +0.0652]        0.354406            0.394636          0.18003
always advance/nearest                   0.060756  [+0.0540, +0.0677]        0.398467            0.331418              NaN
best fixed in hindsight (advance_focus)  0.056907  [+0.0496, +0.0643]        0.346743            0.417625              NaN
random (expected)                        0.077067  [+0.0719, +0.0823]             NaN                 NaN              0.0
```

## S2 (mixed model, exp08_mixed_typed/cf_*)

Largest state difference across a seed's branches at the branch moment: 0.00e+00.

### S2 at 1 s

598 states; mean spread (best − worst branch) 0.0858; all five tied in 1%.
Model − always advance/nearest, regret: +0.0126 [+0.0104, +0.0148].
Model's picks: advance_nearest 0.01, advance_focus 0.58, hold_nearest 0.00, hold_focus 0.39, withdraw_nearest 0.02.

```
                                             regret          regret 95% costly (> 0.05) picks the true best rank correlation
model                                      0.018837  [+0.0171, +0.0207]         0.09699             0.35786         0.297089
always advance/nearest                      0.00626  [+0.0052, +0.0073]         0.01505            0.677258              NaN
best fixed in hindsight (advance_nearest)   0.00626  [+0.0052, +0.0073]         0.01505            0.677258              NaN
random (expected)                          0.026986  [+0.0256, +0.0283]             NaN                 NaN              0.0
```

### S2 at 2 s

598 states; mean spread (best − worst branch) 0.1480; all five tied in 0%.
Model − always advance/nearest, regret: +0.0175 [+0.0138, +0.0213].
Model's picks: advance_nearest 0.01, advance_focus 0.56, hold_nearest 0.00, hold_focus 0.40, withdraw_nearest 0.03.

```
                                             regret          regret 95% costly (> 0.05) picks the true best rank correlation
model                                      0.029689  [+0.0266, +0.0325]        0.244147            0.356187         0.323988
always advance/nearest                     0.012141  [+0.0105, +0.0139]        0.080268            0.633779              NaN
best fixed in hindsight (advance_nearest)  0.012141  [+0.0103, +0.0139]        0.080268            0.633779              NaN
random (expected)                          0.046339  [+0.0443, +0.0486]             NaN                 NaN              0.0
```

### S2 at 4 s

598 states; mean spread (best − worst branch) 0.2628; all five tied in 0%.
Model − always advance/nearest, regret: +0.0088 [+0.0016, +0.0158].
Model's picks: advance_nearest 0.02, advance_focus 0.52, hold_nearest 0.01, hold_focus 0.38, withdraw_nearest 0.08.

```
                                             regret          regret 95% costly (> 0.05) picks the true best rank correlation
model                                      0.044352  [+0.0396, +0.0492]         0.33612            0.429766         0.347656
always advance/nearest                     0.035525  [+0.0312, +0.0399]        0.270903            0.513378              NaN
best fixed in hindsight (advance_nearest)  0.035525  [+0.0312, +0.0399]        0.270903            0.513378              NaN
random (expected)                          0.083507  [+0.0798, +0.0870]             NaN                 NaN              0.0
```

### S2 at 8 s

598 states; mean spread (best − worst branch) 0.2807; all five tied in 1%.
Model − always advance/nearest, regret: -0.0136 [-0.0254, -0.0019].
Model's picks: advance_nearest 0.03, advance_focus 0.39, hold_nearest 0.03, hold_focus 0.44, withdraw_nearest 0.11.

```
                                           regret          regret 95% costly (> 0.05) picks the true best rank correlation
model                                    0.074742  [+0.0661, +0.0836]        0.406355            0.446488         0.215141
always advance/nearest                   0.088385  [+0.0802, +0.0975]          0.4699            0.331104              NaN
best fixed in hindsight (advance_focus)   0.07417  [+0.0653, +0.0830]        0.397993            0.453177              NaN
random (expected)                        0.109937  [+0.1041, +0.1158]             NaN                 NaN              0.0
```

## Target rule and movement, separately

Mean score differences between branches, true and as the model predicts them; and the model's regret
when it may only pick among the nearest variants and withdraw (regret still against the best of all five).

```
              focus − nearest (advance), true   model  hold − advance (nearest), true  model   model regret, nearest-only picks  always advance/nearest
scenario h                                                                                                                                             
S1       1.0                          -0.0113  0.0019                         -0.0014 -0.0006                            0.0056                  0.0052
         2.0                          -0.0116  0.0089                         -0.0024 -0.0016                            0.0113                  0.0109
         4.0                          -0.0028  0.0323                         -0.0035 -0.0021                            0.0273                  0.0267
         8.0                           0.0038  0.0532                         -0.0011 -0.0008                            0.0618                  0.0608
S2       1.0                          -0.0127  0.0009                         -0.0010 -0.0006                            0.0063                  0.0063
         2.0                          -0.0181  0.0041                         -0.0018 -0.0019                            0.0122                  0.0121
         4.0                          -0.0078  0.0180                         -0.0038 -0.0031                            0.0398                  0.0355
         8.0                           0.0142  0.0411                         -0.0038 -0.0012                            0.0893                  0.0884
```
