# Checkpoint 2: one-step damage matrix with measured reach and allocation (19 Nov)

Script: `experiments/mixed/oracle.py`. Fitted on exp08 `train` (6000 battles); one-step R² on
`val`.
STLSQ threshold 0.15 on standardised features. Units: target units removed per second (health
per type in units of ten full units).

## Kept terms and one-step R² (val)

```
                                                                                                                     kept terms    val R²
library target                                                                                                                           
fire    marine (blue)    F[marine->marine] -0.0715, F[marine->marauder] +0.1114, R[marine] -0.1256, F[marauder->marine] -0.1621  0.537618
        marauder (blue)                             F[marine->marine] +0.0551, R[marine] -0.0575, F[marauder->marauder] -0.1183  0.414985
        marine (red)                                F[marine->marauder] +0.1615, R[marine] -0.1755, F[marauder->marine] -0.1683  0.586152
        marauder (red)                                               F[marine->marauder] -0.0646, F[marauder->marauder] -0.1190  0.472795
share   marine (blue)                                                R[marine]*s[marine] -0.1702, R[marauder]*s[marine] -0.1212  0.541805
        marauder (blue)                                          R[marine]*s[marauder] -0.0528, R[marauder]*s[marauder] -0.0811  0.406558
        marine (red)               R[marine]*s[marine] -0.1732, R[marauder]*s[marine] -0.1410, R[marauder]*s[marine]*wd +0.1850  0.618917
        marauder (red)       R[marine]*s[marauder] -0.0547, R[marauder]*s[marauder] -0.0926, R[marauder]*s[marauder]*wd +0.0974  0.503987
rule    marine (blue)                                                R[marine]*s[marine] -0.1242, R[marauder]*s[marine] -0.1319  0.348756
        marauder (blue)                                                                         R[marauder]*s[marauder] -0.1216  0.210137
        marine (red)               R[marine]*s[marine] -0.1541, R[marauder]*s[marine] -0.1898, R[marauder]*s[marine]*wd +0.2281  0.470850
        marauder (red)       R[marine]*s[marauder] -0.0521, R[marauder]*s[marauder] -0.0952, R[marauder]*s[marauder]*wd +0.0999  0.361750
```

## The damage matrix against the game

(i) measured fire; (ii) reach × measured allocation share; (iii) reach × proportional rule.
The firing share from step 1b (single-type data): marines 0.79, marauders 0.54.

```
                           game  (i) per firing unit  (i) / game  (ii) per unit in reach  (ii) / game  (ii) / (game x firing share 1b)  (iii) per unit in reach
attacker        target                                                                                                                                         
marine (red)    marine    0.213                0.071       0.335                   0.170        0.798                            1.010                    0.124
                marauder  0.064               -0.000      -0.000                   0.053        0.825                            1.044                   -0.000
marauder (red)  marine    0.198                0.162       0.821                   0.121        0.614                            1.137                    0.132
                marauder  0.135                0.118       0.875                   0.081        0.600                            1.111                    0.122
marine (blue)   marine    0.213               -0.000      -0.000                   0.173        0.812                            1.028                    0.154
                marauder  0.064                0.065       1.010                   0.055        0.855                            1.082                    0.052
marauder (blue) marine    0.198                0.168       0.852                   0.141        0.714                            1.322                    0.190
                marauder  0.135                0.119       0.881                   0.093        0.686                            1.270                    0.095
```

## Does the per-type state help? One-step R² of the *total* health rate (hp/s), val

The team-level fit is A*-2k's health equation on total hp with total reach (measured).

```
      team-level (one k, total hp)  per type (fire)  per type (share)  per type (rule)
side                                                                                  
blue                         0.441            0.442             0.457            0.408
red                          0.558            0.500             0.572            0.564
```

## Alive law per type (A*-2k's form, Hhat from (ii))

Single-type reference: marines p_B 0.66, p_R 0.54 (focus +0.24); marauders 0.63, 0.53 (+0.15).

```
                     p  p_focus  val R² (alive rate)
units                                               
marine (blue)    0.531      NaN                0.108
marauder (blue)  0.406      NaN                0.050
marine (red)     0.453    0.126                0.160
marauder (red)   0.373    0.099                0.095
```

## (ii) coefficients: 20 battle-bootstrap refits

```
                                             value  kept    q2.5   q97.5
marine (blue)   R[marine]*s[marine]        -0.1702   1.0 -0.1717 -0.1694
                R[marauder]*s[marine]      -0.1212   1.0 -0.1222 -0.1205
marauder (blue) R[marine]*s[marauder]      -0.0528   1.0 -0.0533 -0.0521
                R[marauder]*s[marauder]    -0.0811   1.0 -0.0815 -0.0805
marine (red)    R[marine]*s[marine]        -0.1732   1.0 -0.1745 -0.1723
                R[marauder]*s[marine]      -0.1410   1.0 -0.1417 -0.1397
                R[marauder]*s[marine]*wd    0.1850   1.0  0.1825  0.1876
marauder (red)  R[marine]*s[marauder]      -0.0547   1.0 -0.0552 -0.0543
                R[marauder]*s[marauder]    -0.0926   1.0 -0.0931 -0.0921
                R[marauder]*s[marauder]*wd  0.0974   1.0  0.0966  0.0982
```
