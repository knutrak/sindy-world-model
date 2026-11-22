# Closed-loop test of the μ, σ planner: results

Plan: `34_mu_sigma_closed_loop_plan.md` (approved 22 Nov).
Script: `experiments/planning/closed34.py`.
Fresh seeds 21 x00 000 +, paired by seed.

## Wins per policy

```
                        S1     S2     S3     S4     S5
policy                                                
fixed:hold/focus     0.685    NaN  0.610  0.070    NaN
fixed:kite/focus     0.972  0.855  0.990  0.242  0.602
fixed:kite/nearest   0.968  0.895  0.992  0.290  0.615
model:h3             0.652  0.628  0.735  0.070  0.560
musigma:h3           0.675  0.800  0.745  0.200  0.575
fixed:advance/focus    NaN  0.635    NaN    NaN  0.555
```

## Paired comparisons (μ, σ planner − reference)

```
scenario rule      against  difference              95% read-out   n
      S1   C3 kite/nearest      -0.292 [-0.340, -0.247]    loses 400
      S1   C2     model:h3       0.022 [-0.013, +0.058]     ties 400
      S1   C3   hold/focus      -0.010 [-0.045, +0.025]     ties 400
      S2   C1 kite/nearest      -0.095 [-0.130, -0.060]    loses 400
      S2   C2     model:h3       0.172 [+0.125, +0.220]    beats 400
      S3   C3   kite/focus      -0.245 [-0.290, -0.203]    loses 400
      S3   C2     model:h3       0.010 [-0.043, +0.060]     ties 400
      S3   C3   hold/focus       0.135 [+0.090, +0.180]    beats 400
      S4   C1 kite/nearest      -0.090 [-0.135, -0.048]    loses 400
      S4   C2     model:h3       0.130 [+0.092, +0.168]    beats 400
      S5   C1 kite/nearest      -0.040 [-0.065, -0.018]    loses 400
      S5   C2     model:h3       0.015 [-0.005, +0.037]     ties 400
```

**RQ5 statement (C1 beats in at least one of S2, S4, S5 and loses in none): does not hold.**

## Planner diagnostics (means per battle)

```
scenario  pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  near_tie_share  clip_share  switches  wall_s  pick_kite/nearest  pick_kite/focus
      S1                 0.223               0.339              0.079            0.337                  0.021           0.467       0.256     3.618   0.362                NaN              NaN
      S2                 0.082               0.130              0.102            0.231                  0.039           0.322       0.216     6.498   0.964              0.286            0.131
      S3                 0.116               0.125              0.340            0.132                  0.153           0.300       0.108     6.508   1.573              0.096            0.038
      S4                 0.088               0.075              0.103            0.098                  0.114           0.328       0.257     7.115   1.426              0.373            0.148
      S5                 0.071               0.095              0.081            0.169                  0.059           0.270       0.224     6.430   1.041              0.317            0.209
```

## Read-outs against the plan (`34`)

| | rule | result |
|---|---|---|
| G0 | M4 still passes with clipping | passes; the clip binds in 19-25% of val rollouts and 52-62% of 8 s counterfactual rollouts (expected: few) |
| **C1** | μ, σ planner against best fixed kite in S2, S4, S5 | **loses in all three**: S2 −9.5 [−13.0, −6.0], S4 −9.0 [−13.5, −4.8], S5 −4.0 [−6.5, −1.8] win points |
| **C2** | against the frozen planner | **beats in S2 (+17.2) and S4 (+13.0)**; ties in S1, S3, S5 |
| C3 | flagged | S1 loses to kite (−29.2), ties hold/focus; S3 loses to kite (−24.5), beats hold/focus (+13.5) |

**RQ5 statement (planning with the μ, σ world model useful in this testbed): does not hold.**

Facts beside the rules:
- The planner picks a kite intent in 42% (S2), 52% (S4) and 53% (S5) of re-plans, and switches intent 6-7 times per
  battle.
- Top two candidates within 1 hp in 27-47% of re-plans; a clip binds in 11-26% of candidate rollouts.
- Run time: 11.5 min for 10,000 battles (planner 0.4-1.6 s per battle).
