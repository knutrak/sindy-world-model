# Planning at team level only: results

Plan: `42_team_level_plan.md` (approved 23 Nov). Script: `experiments/planning/team42.py`.
Five team-level candidates; scripted Red; `34`'s seeds, 400 per scenario, paired.

## Wins

```
                        S1     S2     S3     S4     S5
policy                                                
fixed:hold/focus     0.685    NaN  0.610  0.070    NaN
model:h3             0.652  0.628  0.735  0.070  0.560
musigma5:h3          0.675  0.672  0.708  0.102  0.568
oracle5-h3           0.935  0.912  0.930  0.315  0.638
oracle5-h8           0.945  0.955  0.988  0.640  0.805
fixed:advance/focus    NaN  0.635    NaN    NaN  0.555
```

## Paired comparisons

```
scenario    flag rule      policy       against  difference              95% read-out   n  share of 3 s ceiling (T1 / T3)
      S1           T1 musigma5:h3    hold/focus      -0.010 [-0.045, +0.025]     ties 400                             NaN
      S1           T2 musigma5:h3      model:h3       0.022 [-0.013, +0.058]     ties 400                             NaN
      S1           T3  oracle5-h3    hold/focus       0.250 [+0.207, +0.292]    beats 400                             NaN
      S1           T4  oracle5-h8    hold/focus       0.260 [+0.215, +0.305]    beats 400                          -0.040
      S2           T1 musigma5:h3 advance/focus       0.038 [-0.007, +0.083]     ties 400                             NaN
      S2           T2 musigma5:h3      model:h3       0.045 [+0.000, +0.090]     ties 400                             NaN
      S2           T3  oracle5-h3 advance/focus       0.278 [+0.233, +0.323]    beats 400                             NaN
      S2           T4  oracle5-h8 advance/focus       0.320 [+0.275, +0.367]    beats 400                           0.135
      S3 flagged   T1 musigma5:h3    hold/focus       0.098 [+0.052, +0.140]    beats 400                             NaN
      S3 flagged   T2 musigma5:h3      model:h3      -0.028 [-0.080, +0.025]     ties 400                             NaN
      S3 flagged   T3  oracle5-h3    hold/focus       0.320 [+0.275, +0.367]    beats 400                             NaN
      S3 flagged   T4  oracle5-h8    hold/focus       0.378 [+0.333, +0.427]    beats 400                           0.305
      S4 flagged   T1 musigma5:h3    hold/focus       0.032 [+0.003, +0.062]    beats 400                             NaN
      S4 flagged   T2 musigma5:h3      model:h3       0.032 [+0.003, +0.062]    beats 400                             NaN
      S4 flagged   T3  oracle5-h3    hold/focus       0.245 [+0.203, +0.287]    beats 400                             NaN
      S4 flagged   T4  oracle5-h8    hold/focus       0.570 [+0.522, +0.620]    beats 400                           0.133
      S5           T1 musigma5:h3 advance/focus       0.012 [-0.007, +0.033]     ties 400                             NaN
      S5           T2 musigma5:h3      model:h3       0.008 [-0.010, +0.025]     ties 400                             NaN
      S5           T3  oracle5-h3 advance/focus       0.082 [+0.058, +0.110]    beats 400                             NaN
      S5           T4  oracle5-h8 advance/focus       0.250 [+0.210, +0.292]    beats 400                           0.152
```

**Statement (T1 beats in at least one of S1, S2, S5 and loses in none): does not hold.**

## Read-outs against the plan (`42`)

| | rule | result |
|---|---|---|
| **T1** | μ, σ planner (5) − best fixed team intent | **ties in S1, S2, S5** (−1.0, +3.8, +1.2); beats in S3 (+9.8) and S4 (+3.2), both flagged |
| T2 | − frozen planner | ties S1, S2, S3, S5; beats S4 (+3.2) |
| **T3** | 3 s oracle (5) − best fixed team intent | **beats everywhere**: +25.0, +27.8, +32.0, +24.5, +8.2 |
| T4 | 8 s oracle (5) − best fixed | +26.0, +32.0, +37.8, +57.0, +25.0 |
| share | (μ, σ − best fixed) / (3 s oracle − best fixed) | S1 −0.04, S2 0.14, S3 0.31, S4 0.13, S5 0.15 |

**Statement (T1 beats in at least one of S1, S2, S5 and loses in none): does not hold.**

Facts beside the rules:
- At team level the matched ceiling is large: a perfect model with the planners' settings wins 91-94% in S1-S3
  (best fixed 61-69%), and the μ, σ planner captures 0-31% of that gap.
- Run time: 53 min for 6,000 battles.
- (The share column in the table above is printed on the T4 row of each scenario; it is T1 / T3.)
