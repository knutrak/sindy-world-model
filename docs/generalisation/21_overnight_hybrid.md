# Overnight: fixes S, R, T for the unit-level hybrid model: results

Plan: `20_overnight_hybrid_plan.md` (approved 23 Nov).
Script: `experiments/generalisation/overnight20.py`.
Tables are appended as each stage finishes.

## H1 / H2: error ratio to the frozen model (lower is better; H1 passes at ≤ 0.7 for reach)

```
quantity         margin                reach              
h                   1.0    2.0    4.0    1.0    2.0    4.0
scenario variant                                          
marines  R        1.003  1.034  1.011  0.982  1.023  0.916
         RT       1.012  1.041  1.020  0.972  1.001  0.924
         S        1.081  1.286  1.491  1.642  1.535  1.514
         SR       0.992  1.108  1.184  1.013  1.179  1.210
         SRT      0.999  1.106  1.191  1.012  1.176  1.221
         ST       1.087  1.294  1.493  1.641  1.535  1.514
         T        1.035  1.067  1.046  1.545  1.269  1.070
         base     1.033  1.062  1.035  1.548  1.270  1.070
mixed    R        0.996  1.116  1.266  0.946  1.032  1.113
         RT       0.987  1.112  1.265  0.947  1.026  1.120
         S        1.021  1.130  1.169  1.467  1.353  1.338
         SR       0.989  1.085  1.144  1.014  1.163  1.261
         SRT      0.970  1.076  1.145  1.015  1.159  1.257
         ST       1.000  1.127  1.189  1.467  1.352  1.337
         T        1.049  1.225  1.432  1.595  1.425  1.339
         base     1.055  1.227  1.429  1.596  1.425  1.336
```

**Selection:** no variant passes H1 in marines; the run stops here (plan `20`).

Run time: 23 min.


## Read-outs (facts)

- **No variant passes H1** (reach error ≤ 0.7 × the closure's at 2 s and 4 s, marines); per the plan, H3 and H4 were
  not run.
- **Steering (R) is the decisive fix:** with SMAClite's own collision avoidance, marines' reach error is 0.98 ×,
  1.02 × and 0.92 × the closure's at 1, 2, 4 s (base 1.55 / 1.27 / 1.07), and the margin error 1.00-1.03 × frozen.
  Mixed: reach 0.95 / 1.03 / 1.11 ×.
- Target stickiness (T) changes almost nothing; the sequence switch (S) makes reach worse (1.51-1.64 × in marines),
  also when combined with R (1.01-1.21 ×).
- So the unit-level model with steering is as accurate as the closure, not better; the remaining reach error lies
  in the per-unit walk/stop and targeting decisions.
- Run time: 23 min.
