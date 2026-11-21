# Does the model planner withdraw where the oracle does?

Script: `experiments/planning/model_withdraw.py`. Phase B's primary planner replayed (deterministic) on
Phase B's first 200 seeds of S3 and S1; the oracle's decisions from `19_oracle_withdraw.md`.
Exploratory.

## S3

Withdraw share: planner 0.207 of re-plans, oracle 0.243. Predicted gap best − second when the planner withdraws: median 0.0100 (all re-plans 0.0000).

Median state at withdraw and other decisions:

```
                     planner: withdraw  planner: other  oracle: withdraw  oracle: other
t                               10.000          22.000             3.000          9.000
distance                         8.608           8.720             7.377          6.270
in_reach_share_blue              0.000           0.000             0.310          0.714
in_reach_share_red               0.000           0.000             0.154          0.429
depth_red                        0.614           0.520             0.634          0.417
hp_balance                       0.542           0.576             0.598          0.672
unit_balance                     0.333           0.350             0.364          0.438
wounded_share_blue               0.000           0.000             0.000          0.143
wounded_share_red                0.000           0.077             0.000          0.125
```

The oracle's depth-3 rule (fitted on the oracle's decisions) applied to the planner's states:

```
planner withdraws            False  True 
oracle's rule says withdraw              
False                        0.853  0.147
True                         0.579  0.421
```

Battles by whether the planner withdrew in the first 4 s (wins in Phase B):

```
                           battles  planner won  hold/focus won  planner − fixed
choice                                                                          
did not                         21        0.810           0.762            0.048
withdrew in the first 4 s      179        0.749           0.570            0.179
```

## S1

Withdraw share: planner 0.066 of re-plans, oracle 0.228. Predicted gap best − second when the planner withdraws: median 0.0052 (all re-plans 0.0000).

Median state at withdraw and other decisions:

```
                     planner: withdraw  planner: other  oracle: withdraw  oracle: other
t                               24.000           4.000             3.000          5.000
distance                         7.706           5.633             7.011          5.948
in_reach_share_blue              0.000           0.833             0.200          0.714
in_reach_share_red               0.000           0.875             0.200          0.583
depth_red                        0.338           0.275             0.583          0.359
hp_balance                       0.378           0.506             0.501          0.541
unit_balance                     0.300           0.519             0.500          0.556
wounded_share_blue               0.000           0.091             0.000          0.125
wounded_share_red                0.286           0.000             0.000          0.111
```

The oracle's depth-3 rule (fitted on the oracle's decisions) applied to the planner's states:

```
planner withdraws            False  True 
oracle's rule says withdraw              
False                        0.973  0.027
True                         0.889  0.111
```

Battles by whether the planner withdrew in the first 4 s (wins in Phase B):

```
                           battles  planner won  hold/focus won  planner − fixed
choice                                                                          
did not                        194        0.665           0.722           -0.057
withdrew in the first 4 s        6        0.167           0.167            0.000
```

