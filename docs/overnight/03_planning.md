# Planning with the finalists (seeds 50 x00 000 +, 400 battles)

Plan: `00_plan.md`. Script: `experiments/overnight/plan.py`.

## Rules

- modelB1:h3 L1 against the frozen planner: beats 1, ties 4, loses 0 -> not met
- modelB1:h3 L2 against the best fixed intent: beats 1, ties 3, loses 1 -> not met
- modelH5:h3 L1 against the frozen planner: beats 1, ties 4, loses 0 -> not met
- modelH5:h3 L2 against the best fixed intent: beats 1, ties 4, loses 0 -> not met
- modelI7:h3 L1 against the frozen planner: beats 0, ties 5, loses 0 -> not met
- modelI7:h3 L2 against the best fixed intent: beats 1, ties 3, loses 1 -> not met

## Wins

```
                        S1     S2     S3     S4     S5
policy                                                
fixed:hold/focus     0.680    NaN  0.610  0.108    NaN
model:h3             0.640  0.638  0.752  0.118  0.518
modelB1:h3           0.645  0.635  0.832  0.110  0.515
modelH5:h3           0.652  0.652  0.805  0.098  0.505
modelI7:h3           0.638  0.640  0.718  0.112  0.515
oracle5-h3           0.940  0.908  0.932  0.302  0.592
fixed:advance/focus    NaN  0.640    NaN    NaN  0.510
```

## Paired comparisons

```
scenario rule     policy             against  difference              95% read-out   n   share of perfect model's lead
      S1   L1 modelB1:h3            model:h3       0.005 [-0.007, +0.020]     ties 400                             NaN
      S1   L2 modelB1:h3    fixed:hold/focus      -0.035 [-0.068, -0.003]    loses 400                             NaN
      S1   L1 modelH5:h3            model:h3       0.012 [-0.007, +0.030]     ties 400                             NaN
      S1   L2 modelH5:h3    fixed:hold/focus      -0.028 [-0.060, +0.003]     ties 400                             NaN
      S1   L1 modelI7:h3            model:h3      -0.002 [-0.007, +0.000]     ties 400                             NaN
      S1   L2 modelI7:h3    fixed:hold/focus      -0.042 [-0.075, -0.010]    loses 400                             NaN
      S1   L1 oracle5-h3            model:h3       0.300 [+0.255, +0.347]    beats 400                             NaN
      S1   L2 oracle5-h3    fixed:hold/focus       0.260 [+0.217, +0.305]    beats 400 B1: -0.13, H5: -0.11, I7: -0.16
      S2   L1 modelB1:h3            model:h3      -0.002 [-0.015, +0.010]     ties 400                             NaN
      S2   L2 modelB1:h3 fixed:advance/focus      -0.005 [-0.020, +0.010]     ties 400                             NaN
      S2   L1 modelH5:h3            model:h3       0.015 [-0.013, +0.045]     ties 400                             NaN
      S2   L2 modelH5:h3 fixed:advance/focus       0.012 [-0.013, +0.037]     ties 400                             NaN
      S2   L1 modelI7:h3            model:h3       0.002 [-0.013, +0.020]     ties 400                             NaN
      S2   L2 modelI7:h3 fixed:advance/focus       0.000 [-0.010, +0.010]     ties 400                             NaN
      S2   L1 oracle5-h3            model:h3       0.270 [+0.228, +0.315]    beats 400                             NaN
      S2   L2 oracle5-h3 fixed:advance/focus       0.268 [+0.223, +0.310]    beats 400 B1: -0.02, H5: +0.05, I7: +0.00
      S3   L1 modelB1:h3            model:h3       0.080 [+0.037, +0.122]    beats 400                             NaN
      S3   L2 modelB1:h3    fixed:hold/focus       0.222 [+0.177, +0.270]    beats 400                             NaN
      S3   L1 modelH5:h3            model:h3       0.052 [+0.013, +0.092]    beats 400                             NaN
      S3   L2 modelH5:h3    fixed:hold/focus       0.195 [+0.147, +0.245]    beats 400                             NaN
      S3   L1 modelI7:h3            model:h3      -0.035 [-0.087, +0.015]     ties 400                             NaN
      S3   L2 modelI7:h3    fixed:hold/focus       0.108 [+0.070, +0.147]    beats 400                             NaN
      S3   L1 oracle5-h3            model:h3       0.180 [+0.138, +0.225]    beats 400                             NaN
      S3   L2 oracle5-h3    fixed:hold/focus       0.322 [+0.275, +0.372]    beats 400 B1: +0.69, H5: +0.60, I7: +0.33
      S4   L1 modelB1:h3            model:h3      -0.008 [-0.018, +0.000]     ties 400                             NaN
      S4   L2 modelB1:h3    fixed:hold/focus       0.002 [-0.020, +0.025]     ties 400                             NaN
      S4   L1 modelH5:h3            model:h3      -0.020 [-0.040, +0.000]     ties 400                             NaN
      S4   L2 modelH5:h3    fixed:hold/focus      -0.010 [-0.033, +0.013]     ties 400                             NaN
      S4   L1 modelI7:h3            model:h3      -0.005 [-0.015, +0.005]     ties 400                             NaN
      S4   L2 modelI7:h3    fixed:hold/focus       0.005 [-0.018, +0.028]     ties 400                             NaN
      S4   L1 oracle5-h3            model:h3       0.185 [+0.142, +0.228]    beats 400                             NaN
      S4   L2 oracle5-h3    fixed:hold/focus       0.195 [+0.152, +0.237]    beats 400 B1: +0.01, H5: -0.05, I7: +0.03
      S5   L1 modelB1:h3            model:h3      -0.002 [-0.007, +0.000]     ties 400                             NaN
      S5   L2 modelB1:h3 fixed:advance/focus       0.005 [-0.005, +0.015]     ties 400                             NaN
      S5   L1 modelH5:h3            model:h3      -0.012 [-0.030, +0.003]     ties 400                             NaN
      S5   L2 modelH5:h3 fixed:advance/focus      -0.005 [-0.022, +0.013]     ties 400                             NaN
      S5   L1 modelI7:h3            model:h3      -0.002 [-0.007, +0.000]     ties 400                             NaN
      S5   L2 modelI7:h3 fixed:advance/focus       0.005 [+0.000, +0.013]     ties 400                             NaN
      S5   L1 oracle5-h3            model:h3       0.075 [+0.048, +0.105]    beats 400                             NaN
      S5   L2 oracle5-h3 fixed:advance/focus       0.082 [+0.055, +0.113]    beats 400 B1: +0.06, H5: -0.06, I7: +0.06
```
