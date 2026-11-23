# Perfect geometry, SINDy damage: results

Plan: `50_geometry_oracle_plan.md`. Script: `experiments/planning/geo50.py`.
`34`'s first 200 seeds per scenario, paired.

```
scenario     policy       against  wins  its wins  difference              95%
      S1    geo5:h3    hold/focus 0.570     0.675      -0.105 [-0.180, -0.030]
      S1 oracle5-h3    hold/focus 0.935     0.675       0.260 [+0.200, +0.320]
      S1    geo5:h3    oracle5-h3 0.570     0.935      -0.365 [-0.430, -0.300]
      S1    geo5:h3   musigma5:h3 0.570     0.655      -0.085 [-0.170, +0.000]
      S2    geo5:h3 advance/focus 0.380     0.615      -0.235 [-0.320, -0.150]
      S2 oracle5-h3 advance/focus 0.915     0.615       0.300 [+0.235, +0.370]
      S2    geo5:h3    oracle5-h3 0.380     0.915      -0.535 [-0.610, -0.460]
      S2    geo5:h3   musigma5:h3 0.380     0.665      -0.285 [-0.365, -0.200]
      S3    geo5:h3    hold/focus 0.490     0.630      -0.140 [-0.225, -0.055]
      S3 oracle5-h3    hold/focus 0.930     0.630       0.300 [+0.240, +0.365]
      S3    geo5:h3    oracle5-h3 0.490     0.930      -0.440 [-0.510, -0.370]
      S3    geo5:h3   musigma5:h3 0.490     0.725      -0.235 [-0.330, -0.140]
      S4    geo5:h3    hold/focus 0.085     0.065       0.020 [-0.025, +0.065]
      S4 oracle5-h3    hold/focus 0.285     0.065       0.220 [+0.165, +0.280]
      S4    geo5:h3    oracle5-h3 0.085     0.285      -0.200 [-0.265, -0.140]
      S4    geo5:h3   musigma5:h3 0.085     0.110      -0.025 [-0.070, +0.020]
      S5    geo5:h3 advance/focus 0.360     0.545      -0.185 [-0.250, -0.120]
      S5 oracle5-h3 advance/focus 0.645     0.545       0.100 [+0.060, +0.145]
      S5    geo5:h3    oracle5-h3 0.360     0.645      -0.285 [-0.350, -0.225]
      S5    geo5:h3   musigma5:h3 0.360     0.565      -0.205 [-0.270, -0.135]
```

Share of the matched ceiling captured, (geo5 − best fixed) / (oracle5-h3 − best fixed):

```
S1   -0.40
S2   -0.78
S3   -0.47
S4    0.09
S5   -1.85
```

**Read-out fixed in the plan: damage and death equations are a second bottleneck** (share ≥ 0.7 in 0 of 5 scenarios, ≤ 0.3 in 5).

## Read-outs and a check (facts)

- **The geometry oracle loses to the best fixed team intent** in S1 (−10.5), S2 (−23.5), S3 (−14.0) and S5 (−18.5),
  ties in S4 (+2.0), and is below the μ, σ planner in S2, S3 and S5. By the fixed read-out: *damage and death equations
  are a second bottleneck* (share of the matched ceiling ≤ 0.3 in all five scenarios).
- **Check that the method works** (`experiments/planning/geo50_check.py`, 16 battles of S1 and S2, the exact oracle
  playing, both scoring the same states): the geometry oracle's candidate scores correlate with the exact oracle's
  true scores at a median of 0.96 in both scenarios, and it picks the same candidate in 56% of 154 re-plans (the
  world-model planners: 27-32%, `45`). Its main errors are nearest against focus, and withdraw where the exact oracle
  advances (12 of 85).
- So the scores are mostly right in level, but the *differences* between candidates are small, and the model's
  damage and death predictions flip enough of them to lose battles.

**Caveat (stated in the plan, and likely to matter):** the test combines the game's true reach with the model's own
health and unit counts. When the model's units die faster or slower than the game's, the two are inconsistent (for
example, more units in reach than the model has alive), which can distort the predicted damage. So the result says
"perfect reach alone, plugged into the SINDy damage law, does not make a good planner", not that a consistent
unit-level model with correct geometry would fail.
