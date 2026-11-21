# The perturbed oracle with shared noise

Script: `experiments/planning/shared_noise.py`. Phase A's perturbed oracle (positions ± 0.2, health ± 5%)
with one draw per re-plan applied to all five candidates' copies, instead of one draw per copy; Phase A's
200 oracle seeds per scenario, paired.

## Wins

```
                           shared won  against won  difference               95%    n  share of exact-oracle headroom
scenario against                                                                                                     
S1       oracle-perturbed       0.860        0.865      -0.005  [-0.055, +0.045]  200                             NaN
         oracle                 0.860        0.940      -0.080  [-0.135, -0.030]  200                             NaN
         hold/focus             0.860        0.700       0.160  [+0.095, +0.225]  200                           0.667
S2       oracle-perturbed       0.860        0.860       0.000  [-0.050, +0.050]  200                             NaN
         oracle                 0.860        0.965      -0.105  [-0.155, -0.060]  200                             NaN
         advance/focus          0.860        0.645       0.215  [+0.155, +0.280]  200                           0.672
S3       oracle-perturbed       0.950        0.935       0.015  [-0.010, +0.045]  200                             NaN
         oracle                 0.950        0.980      -0.030  [-0.060, -0.005]  200                             NaN
         hold/focus             0.950        0.560       0.390  [+0.325, +0.460]  200                           0.929
S4       oracle-perturbed       0.525        0.465       0.060  [-0.010, +0.135]  200                             NaN
         oracle                 0.525        0.650      -0.125  [-0.185, -0.060]  200                             NaN
         hold/focus             0.525        0.100       0.425  [+0.350, +0.495]  200                           0.773
S5       oracle-perturbed       0.740        0.705       0.035  [-0.010, +0.080]  200                             NaN
         oracle                 0.740        0.790      -0.050  [-0.095, -0.010]  200                             NaN
         advance/focus          0.740        0.560       0.180  [+0.125, +0.235]  200                           0.783
```

## Intent use (share of steps) and how close the top two candidates are

```
          advance/nearest  advance/focus  withdraw/nearest  hold/focus  hold/nearest  top two within 1 hp  median gap (hp)
scenario                                                                                                                  
S1                  0.387          0.315             0.234       0.021         0.043                0.685              0.0
S2                  0.441          0.303             0.208       0.013         0.036                0.729              0.0
S3                  0.364          0.254             0.317       0.009         0.056                0.606              0.0
S4                  0.410          0.234             0.280       0.011         0.065                0.636              0.0
S5                  0.446          0.286             0.217       0.013         0.037                0.715              0.0
```

