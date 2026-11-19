# Checkpoint 2: one-step damage matrix with measured reach and allocation (19 Nov)

Script: `experiments/mixed/oracle.py`. Fitted on exp08 `train` (6000 battles); one-step R² on
`val`.
STLSQ threshold 0.15 on standardised features. Units: target units removed per second (health
per type in units of ten full units).

## Kept terms and one-step R² (val)

```
                                                                                                                                            kept terms    val R²
library target                                                                                                                                                  
fire    marine (blue)    F[marine->marine] -0.0876, F[marine->marauder] +0.0994, R[marine] -0.1026, F[marauder->marauder] +0.1311, R[marauder] -0.1448  0.557723
        marauder (blue)  F[marine->marine] +0.0571, R[marine] -0.0514, F[marauder->marauder] -0.0647, F[marauder->marine] +0.0492, R[marauder] -0.0556  0.438653
        marine (red)     F[marine->marine] -0.1088, F[marine->marauder] +0.0778, R[marine] -0.0896, F[marauder->marauder] +0.1306, R[marauder] -0.1345  0.600477
        marauder (red)                                                                      F[marine->marauder] -0.0646, F[marauder->marauder] -0.1190  0.472795
share   marine (blue)                                                                       R[marine]*s[marine] -0.1784, R[marauder]*s[marine] -0.1651  0.554476
        marauder (blue)                                                                 R[marine]*s[marauder] -0.0600, R[marauder]*s[marauder] -0.1127  0.424617
        marine (red)                                                                        R[marine]*s[marine] -0.1845, R[marauder]*s[marine] -0.1626  0.593601
        marauder (red)                                                                  R[marine]*s[marauder] -0.0596, R[marauder]*s[marauder] -0.1125  0.490559
rule    marine (blue)                                                                       R[marine]*s[marine] -0.1404, R[marauder]*s[marine] -0.1754  0.370174
        marauder (blue)                                                                 R[marine]*s[marauder] -0.0891, R[marauder]*s[marauder] -0.1176  0.262002
        marine (red)                                                                        R[marine]*s[marine] -0.1862, R[marauder]*s[marine] -0.1859  0.437410
        marauder (red)                                                                  R[marine]*s[marauder] -0.0626, R[marauder]*s[marauder] -0.1075  0.341173
```

## The damage matrix against the game

(i) measured fire; (ii) reach × measured allocation share; (iii) reach × proportional rule.
The firing share from step 1b (single-type data): marines 0.79, marauders 0.54.

```
                           game  (i) per firing unit  (i) / game  (ii) per unit in reach  (ii) / game  (ii) / (game x firing share 1b)  (iii) per unit in reach
attacker        target                                                                                                                                         
marine (red)    marine    0.213                0.088       0.410                   0.178        0.836                            1.059                    0.140
                marauder  0.064               -0.000      -0.000                   0.060        0.937                            1.186                    0.089
marauder (red)  marine    0.198               -0.000      -0.000                   0.165        0.836                            1.548                    0.175
                marauder  0.135                0.065       0.479                   0.113        0.834                            1.545                    0.118
marine (blue)   marine    0.213                0.109       0.510                   0.184        0.865                            1.095                    0.186
                marauder  0.064                0.065       1.010                   0.060        0.932                            1.180                    0.063
marauder (blue) marine    0.198               -0.000      -0.000                   0.163        0.823                            1.524                    0.186
                marauder  0.135                0.119       0.881                   0.112        0.832                            1.541                    0.108
```

## Does the per-type state help? One-step R² of the *total* health rate (hp/s), val

The team-level fit is A*-2k's health equation on total hp with total reach (measured).

```
      team-level (one k, total hp)  per type (fire)  per type (share)  per type (rule)
side                                                                                  
blue                         0.453            0.490             0.484            0.465
red                          0.545            0.517             0.527            0.515
```

## Alive law per type (A*-2k's form, Hhat from (ii))

Single-type reference: marines p_B 0.66, p_R 0.54 (focus +0.24); marauders 0.63, 0.53 (+0.15).

```
                     p  p_focus  val R² (alive rate)
units                                               
marine (blue)    0.521      NaN                0.113
marauder (blue)  0.392      NaN                0.053
marine (red)     0.447    0.124                0.148
marauder (red)   0.350    0.088                0.083
```

## (ii) coefficients: 20 battle-bootstrap refits

```
                                          value  kept    q2.5   q97.5
marine (blue)   R[marine]*s[marine]     -0.1784   1.0 -0.1801 -0.1778
                R[marauder]*s[marine]   -0.1651   1.0 -0.1661 -0.1638
marauder (blue) R[marine]*s[marauder]   -0.0600   1.0 -0.0605 -0.0593
                R[marauder]*s[marauder] -0.1127   1.0 -0.1132 -0.1123
marine (red)    R[marine]*s[marine]     -0.1845   1.0 -0.1858 -0.1835
                R[marauder]*s[marine]   -0.1626   1.0 -0.1634 -0.1609
marauder (red)  R[marine]*s[marauder]   -0.0596   1.0 -0.0600 -0.0593
                R[marauder]*s[marauder] -0.1125   1.0 -0.1131 -0.1120
```
