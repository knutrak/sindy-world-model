# Step 4: counterfactual intent ranking on mixed battles (exp08 cf_*, 19 Nov)

Script: `experiments/mixed/cf_ranking.py`. Branches: ['advance_focus', 'advance_nearest', 'hold_focus', 'hold_nearest', 'withdraw_focus', 'withdraw_nearest'].

## At 4 s (598 seeds; mean spread of the true margin change across branches 0.114; best fixed intent in hindsight: advance_nearest)

```
                        rank correlation  picks the true best  regret
team A*-2k                         0.535                0.421   0.042
typed (closure)                    0.509                0.415   0.045
typed (rule)                       0.511                0.413   0.045
always advance_nearest               NaN                0.513   0.036
random                             0.000                0.314   0.121
```

## At 8 s (598 seeds; mean spread of the true margin change across branches 0.121; best fixed intent in hindsight: advance_focus)

```
                      rank correlation  picks the true best  regret
team A*-2k                       0.269                0.408   0.080
typed (closure)                  0.257                0.396   0.083
typed (rule)                     0.245                0.393   0.082
always advance_focus               NaN                0.453   0.074
random                           0.000                0.313   0.138
```
