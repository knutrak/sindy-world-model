# When does the exact oracle withdraw?

Script: `experiments/planning/oracle_withdraw.py`. Phase A's exact oracle replayed (deterministic) on
Phase A's first 100 seeds of S1, S2, S3, S5, with the state at each re-plan (1 s). Exploratory.
Reach shares: share of a side's alive units with an enemy in reach (capped at 6); reach asymmetry = Blue's
share − Red's. Health balance = Blue hp / (Blue + Red hp). Wounded: below half of own start health.

## Overview

```
          decisions  withdraw share  battles with any withdraw   won
scenario                                                            
S1              965           0.228                       0.68  0.95
S2             1266           0.156                       0.79  0.97
S3             1630           0.243                       0.95  0.98
S5             1551           0.188                       0.86  0.82
```

## S1

Median state when the oracle withdraws vs when it does not:

```
                     withdraw  other  std. difference
hp_balance              0.501  0.541           -0.594
unit_balance            0.500  0.556           -0.573
reach_asymmetry         0.000  0.000           -0.095
in_reach_share_blue     0.200  0.714           -0.855
in_reach_share_red      0.200  0.583           -0.800
mean_health_blue        0.649  0.598            0.421
mean_health_red         0.693  0.613            0.523
wounded_share_blue      0.000  0.125           -0.430
wounded_share_red       0.000  0.111           -0.506
distance                7.011  5.948            0.713
front_distance          5.513  5.301            0.477
depth_red               0.583  0.359            0.785
width_red               0.656  0.653            0.060
t                       3.000  5.000           -0.473
```

Withdraw spells: 114; length in re-plans (1 s): median 2, share of length 1: 49%, longest 6.
Score gap when it withdraws (withdraw − best other, hp): median 24.2; as a share of both sides' hp: median 0.046.

Depth-3 tree, withdraw vs not: balanced accuracy 0.700 (5-fold by battle; 0.5 = chance).

```
|--- in_reach_share_blue <= 0.41
|   |--- t <= 7.50
|   |   |--- distance <= 7.63
|   |   |   |--- class: True
|   |   |--- distance >  7.63
|   |   |   |--- class: True
|   |--- t >  7.50
|   |   |--- class: True
|--- in_reach_share_blue >  0.41
|   |--- hp_balance <= 0.56
|   |   |--- depth_red <= 0.33
|   |   |   |--- class: False
|   |   |--- depth_red >  0.33
|   |   |   |--- class: True
|   |--- hp_balance >  0.56
|   |   |--- distance <= 5.77
|   |   |   |--- class: False
|   |   |--- distance >  5.77
|   |   |   |--- class: False

```

## S2

Median state when the oracle withdraws vs when it does not:

```
                     withdraw  other  std. difference
hp_balance              0.502  0.537           -0.544
unit_balance            0.500  0.562           -0.631
reach_asymmetry         0.000  0.000            0.059
in_reach_share_blue     0.348  0.750           -0.743
in_reach_share_red      0.286  0.667           -0.774
mean_health_blue        0.693  0.591            0.564
mean_health_red         0.663  0.581            0.532
wounded_share_blue      0.000  0.182           -0.543
wounded_share_red       0.000  0.167           -0.528
distance                7.038  6.068            0.760
front_distance          5.559  5.424            0.207
depth_red               0.711  0.449            0.874
width_red               0.948  0.898            0.158
t                       2.000  6.000           -0.632
```

Withdraw spells: 128; length in re-plans (1 s): median 1, share of length 1: 62%, longest 6.
Score gap when it withdraws (withdraw − best other, hp): median 33.1; as a share of both sides' hp: median 0.040.

Depth-3 tree, withdraw vs not: balanced accuracy 0.665 (5-fold by battle; 0.5 = chance).

```
|--- depth_red <= 0.50
|   |--- distance <= 5.98
|   |   |--- unit_balance <= 0.51
|   |   |   |--- class: False
|   |   |--- unit_balance >  0.51
|   |   |   |--- class: False
|   |--- distance >  5.98
|   |   |--- mean_health_red <= 0.48
|   |   |   |--- class: False
|   |   |--- mean_health_red >  0.48
|   |   |   |--- class: False
|--- depth_red >  0.50
|   |--- distance <= 6.94
|   |   |--- width_red <= 1.31
|   |   |   |--- class: True
|   |   |--- width_red >  1.31
|   |   |   |--- class: False
|   |--- distance >  6.94
|   |   |--- width_red <= 0.82
|   |   |   |--- class: True
|   |   |--- width_red >  0.82
|   |   |   |--- class: True

```

## S3

Median state when the oracle withdraws vs when it does not:

```
                     withdraw  other  std. difference
hp_balance              0.598  0.672           -0.541
unit_balance            0.364  0.438           -0.575
reach_asymmetry         0.036  0.133           -0.234
in_reach_share_blue     0.310  0.714           -0.627
in_reach_share_red      0.154  0.429           -0.556
mean_health_blue        0.665  0.589            0.407
mean_health_red         0.705  0.633            0.487
wounded_share_blue      0.000  0.143           -0.405
wounded_share_red       0.000  0.125           -0.503
distance                7.377  6.270            0.645
front_distance          5.749  5.614            0.209
depth_red               0.634  0.417            0.857
width_red               0.626  0.584            0.075
t                       3.000  9.000           -0.691
```

Withdraw spells: 222; length in re-plans (1 s): median 1, share of length 1: 62%, longest 8.
Score gap when it withdraws (withdraw − best other, hp): median 25.2; as a share of both sides' hp: median 0.041.

Depth-3 tree, withdraw vs not: balanced accuracy 0.711 (5-fold by battle; 0.5 = chance).

```
|--- t <= 3.50
|   |--- width_red <= 0.93
|   |   |--- distance <= 6.36
|   |   |   |--- class: True
|   |   |--- distance >  6.36
|   |   |   |--- class: True
|   |--- width_red >  0.93
|   |   |--- reach_asymmetry <= 0.16
|   |   |   |--- class: False
|   |   |--- reach_asymmetry >  0.16
|   |   |   |--- class: False
|--- t >  3.50
|   |--- hp_balance <= 0.62
|   |   |--- depth_red <= 0.67
|   |   |   |--- class: False
|   |   |--- depth_red >  0.67
|   |   |   |--- class: True
|   |--- hp_balance >  0.62
|   |   |--- distance <= 5.82
|   |   |   |--- class: False
|   |   |--- distance >  5.82
|   |   |   |--- class: False

```

## S5

Median state when the oracle withdraws vs when it does not:

```
                     withdraw  other  std. difference
hp_balance              0.426  0.479           -0.190
unit_balance            0.500  0.538           -0.470
reach_asymmetry         0.000  0.000           -0.122
in_reach_share_blue     0.250  0.667           -0.488
in_reach_share_red      0.240  0.500           -0.443
mean_health_blue        0.664  0.561            0.533
mean_health_red         0.658  0.614            0.351
wounded_share_blue      0.000  0.182           -0.477
wounded_share_red       0.000  0.111           -0.420
distance                7.260  6.334            0.634
front_distance          5.664  5.549            0.131
depth_red               0.714  0.448            0.844
width_red               0.855  0.738            0.294
t                       3.000  8.000           -0.509
```

Withdraw spells: 170; length in re-plans (1 s): median 1, share of length 1: 58%, longest 7.
Score gap when it withdraws (withdraw − best other, hp): median 28.4; as a share of both sides' hp: median 0.038.

Depth-3 tree, withdraw vs not: balanced accuracy 0.660 (5-fold by battle; 0.5 = chance).

```
|--- depth_red <= 0.52
|   |--- depth_red <= 0.31
|   |   |--- width_red <= 0.49
|   |   |   |--- class: False
|   |   |--- width_red >  0.49
|   |   |   |--- class: False
|   |--- depth_red >  0.31
|   |   |--- in_reach_share_blue <= 0.76
|   |   |   |--- class: True
|   |   |--- in_reach_share_blue >  0.76
|   |   |   |--- class: False
|--- depth_red >  0.52
|   |--- in_reach_share_blue <= 0.36
|   |   |--- hp_balance <= 0.44
|   |   |   |--- class: True
|   |   |--- hp_balance >  0.44
|   |   |   |--- class: True
|   |--- in_reach_share_blue >  0.36
|   |   |--- front_distance <= 5.28
|   |   |   |--- class: True
|   |   |--- front_distance >  5.28
|   |   |   |--- class: False

```

