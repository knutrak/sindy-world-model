# The confidence-gated planner: results

Plan: `22_gated_planner_plan.md` (pre-registered). Script: `experiments/planning/gated_planner.py`.
Thresholds tau (team margin, ten units of the first type): {"marines": {"tau": 0.03324024725082439, "states": 522, "pairs": 5220, "p25": 0.005093044157318059, "p75": 0.07513196878735498}, "mixed": {"tau": 0.041934076740337334, "states": 598, "pairs": 5980, "p25": 0.006163841575627847, "p75": 0.0907346690209828}}.

## Wins, paired over 400 fresh seeds (18 x00 000 +)

```
                                 arm won  against won  difference               95% read-out  flagged    n
scenario arm      against                                                                                 
S1       gated:h3 hold/focus       0.675        0.690      -0.015  [-0.028, -0.005]    loses    False  400
                  model:h3         0.675        0.655       0.020  [-0.013, +0.052]     ties    False  400
         model:h3 hold/focus       0.655        0.690      -0.035  [-0.070, +0.000]     ties    False  400
S2       gated:h3 advance/focus    0.628        0.622       0.005  [+0.000, +0.013]     ties    False  400
                  model:h3         0.628        0.628       0.000  [-0.018, +0.018]     ties    False  400
         model:h3 advance/focus    0.628        0.622       0.005  [-0.015, +0.025]     ties    False  400
S3       gated:h3 hold/focus       0.672        0.592       0.080  [+0.045, +0.117]    beats     True  400
                  model:h3         0.672        0.708      -0.035  [-0.083, +0.015]     ties     True  400
         model:h3 hold/focus       0.708        0.592       0.115  [+0.058, +0.168]    beats     True  400
S4       gated:h3 hold/focus       0.098        0.100      -0.002  [-0.007, +0.000]     ties     True  400
                  model:h3         0.098        0.098       0.000  [-0.018, +0.018]     ties     True  400
         model:h3 hold/focus       0.098        0.100      -0.002  [-0.020, +0.018]     ties     True  400
S5       gated:h3 advance/focus    0.498        0.498       0.000  [+0.000, +0.000]     ties    False  400
                  model:h3         0.498        0.498       0.000  [-0.013, +0.013]     ties    False  400
         model:h3 advance/focus    0.498        0.498       0.000  [-0.013, +0.013]     ties    False  400
```

## What the gated planner plays (share of steps)

```
                default  share of re-plans off the default  hold/focus  advance/focus  withdraw/nearest  advance/nearest  hold/nearest  withdraw/focus best fixed on these seeds
scenario                                                                                                                                                                        
S1           hold/focus                              0.010       0.989          0.010             0.001              0.0           0.0             0.0          hold/focus 0.690
S2        advance/focus                              0.019       0.000          0.980             0.019              0.0           0.0             0.0       advance/focus 0.623
S3           hold/focus                              0.064       0.934          0.001             0.065              0.0           0.0             0.0          hold/focus 0.593
S4           hold/focus                              0.033       0.965          0.005             0.030              0.0           0.0             0.0       advance/focus 0.100
S5        advance/focus                              0.029       0.000          0.970             0.030              0.0           0.0             0.0     advance/nearest 0.502
```

