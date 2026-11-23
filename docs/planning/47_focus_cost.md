# Focus cost on the μ, σ planner, team level: results

Plan: `46_focus_cost_plan.md`. Script: `experiments/planning/fc46.py`. Fresh seeds 23 x00 000 +.

## Wins

```
                          S1     S2     S3     S4     S5
policy                                                  
fixed:hold/focus       0.668    NaN  0.635  0.085    NaN
musigma5-focuscost:h3  0.678  0.610  0.702  0.092  0.540
musigma5:h3            0.700  0.642  0.730  0.092  0.542
oracle5-h3             0.930  0.892  0.920  0.335  0.648
fixed:advance/focus      NaN  0.600    NaN    NaN  0.538
```

## Comparisons

```
scenario    flag                rule                policy       against  difference              95% read-out     n
      S1                          F1 musigma5-focuscost:h3   musigma5:h3      -0.022 [-0.052, +0.007]     ties 400.0
      S1                          F2 musigma5-focuscost:h3    hold/focus       0.010 [-0.033, +0.052]     ties 400.0
      S1                     ceiling            oracle5-h3    hold/focus       0.262 [+0.220, +0.307]    beats 400.0
      S1     NaN F3 share of ceiling                   NaN           NaN       0.038              NaN      NaN   NaN
      S2                          F1 musigma5-focuscost:h3   musigma5:h3      -0.032 [-0.068, +0.003]     ties 400.0
      S2                          F2 musigma5-focuscost:h3 advance/focus       0.010 [-0.033, +0.055]     ties 400.0
      S2                     ceiling            oracle5-h3 advance/focus       0.292 [+0.247, +0.338]    beats 400.0
      S2     NaN F3 share of ceiling                   NaN           NaN       0.034              NaN      NaN   NaN
      S3 flagged                  F1 musigma5-focuscost:h3   musigma5:h3      -0.028 [-0.048, -0.007]    loses 400.0
      S3 flagged                  F2 musigma5-focuscost:h3    hold/focus       0.068 [+0.025, +0.113]    beats 400.0
      S3 flagged             ceiling            oracle5-h3    hold/focus       0.285 [+0.240, +0.333]    beats 400.0
      S3     NaN F3 share of ceiling                   NaN           NaN       0.237              NaN      NaN   NaN
      S4 flagged                  F1 musigma5-focuscost:h3   musigma5:h3       0.000 [-0.020, +0.020]     ties 400.0
      S4 flagged                  F2 musigma5-focuscost:h3    hold/focus       0.008 [-0.020, +0.037]     ties 400.0
      S4 flagged             ceiling            oracle5-h3    hold/focus       0.250 [+0.207, +0.292]    beats 400.0
      S4     NaN F3 share of ceiling                   NaN           NaN       0.030              NaN      NaN   NaN
      S5                          F1 musigma5-focuscost:h3   musigma5:h3      -0.002 [-0.020, +0.013]     ties 400.0
      S5                          F2 musigma5-focuscost:h3 advance/focus       0.002 [-0.018, +0.022]     ties 400.0
      S5                     ceiling            oracle5-h3 advance/focus       0.110 [+0.080, +0.140]    beats 400.0
      S5     NaN F3 share of ceiling                   NaN           NaN       0.023              NaN      NaN   NaN
```

**Statement (F1 beats in at least one of S1, S2, S5 and loses in none): does not hold.**

## Read-outs against the plan (`46`)

| | rule | result |
|---|---|---|
| **F1** | focus cost − μ, σ planner | **ties in S1, S2, S4, S5** (−2.2, −3.2, 0.0, −0.2); loses in S3 (−2.8, flagged) |
| F2 | focus cost − best fixed team intent | ties S1, S2, S4, S5; beats S3 (+6.8) |
| F3 | share of the matched ceiling | 2-4% (S3 24%) |

**Statement (F1 beats in at least one of S1, S2, S5 and loses in none): does not hold.**

Facts beside the rules:
- The correction does shift the picks towards nearest, as intended: S1 advance/nearest + hold/nearest 31% → 62% of
  re-plans; S4 57% → 78%.
- Better agreement on target choice did not bring wins: what the perfect model gains on must lie elsewhere (when to
  hold, advance or withdraw).
- On fresh seeds the earlier picture replicates: the μ, σ planner (no focus cost) ties the best fixed team intent in
  S1, S2, S5 (70.0 vs 66.8, 64.2 vs 60.0, 54.2 vs 53.8) and gains in S3 (73.0 vs 63.5). The matched ceiling
  (`oracle5-h3`) is +25 to +29 points in S1-S4 and +11 in S5.
- Run time: about 35 min for 8,000 battles.
- A wiring error in the first launch (the variant fell through to the frozen model) was caught by a class check and
  stopped before any results were written; the run reported here uses the μ, σ model (checked).
