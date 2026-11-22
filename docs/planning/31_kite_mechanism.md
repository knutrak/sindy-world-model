# Why kite wins: a mechanism check

Script: `experiments/planning/kite_mechanism.py` (see its docstring). Red is SMAClite's scripted
attack-move or the same kite rule; 'reach' = kite's firing rule (shoot only inside own range) without
the step back. Means over battles.
Duty = shots x weapon cooldown / unit-seconds alive (1 = fires at every chance); 'ready, none in range'
and 'in range' are shares of unit-steps after first contact.

```
                                  won  blue duty  red duty  blue ready, none in range  red ready, none in range  blue in range  red in range  red hp lost  blue hp lost  seconds
scenario red    blue                                                                                                                                                            
S1       a-move advance/focus    0.66      0.724     0.681                      0.233                     0.223          0.744         0.771        0.919         0.823    8.428
                advance/nearest  0.59      0.710     0.643                      0.263                     0.297          0.709         0.698        0.919         0.848    7.864
                reach/focus      0.65      0.718     0.655                      0.257                     0.273          0.722         0.721        0.927         0.813    8.245
                kite/focus       0.97      0.782     0.577                      0.170                     0.320          0.577         0.644        0.997         0.624    6.361
                kite/nearest     0.97      0.797     0.566                      0.157                     0.336          0.578         0.627        0.997         0.622    6.055
         kite   advance/focus    0.34      0.685     0.762                      0.250                     0.117          0.624         0.693        0.821         0.940    8.160
                kite/focus       0.87      0.749     0.619                      0.184                     0.214          0.526         0.546        0.985         0.755    6.636
S2       a-move advance/focus    0.65      0.777     0.737                      0.139                     0.144          0.854         0.851        0.907         0.820   11.851
                advance/nearest  0.54      0.766     0.708                      0.158                     0.170          0.832         0.826        0.903         0.858   11.396
                reach/focus      0.64      0.771     0.721                      0.144                     0.156          0.848         0.840        0.900         0.824   11.860
                kite/focus       0.83      0.811     0.688                      0.085                     0.167          0.736         0.814        0.980         0.740   10.879
                kite/nearest     0.89      0.833     0.677                      0.076                     0.181          0.735         0.802        0.991         0.689    9.841
         kite   advance/focus    0.27      0.691     0.737                      0.176                     0.066          0.778         0.794        0.816         0.951   13.251
                kite/focus       0.86      0.679     0.648                      0.172                     0.110          0.638         0.763        0.979         0.748   12.619
S3       a-move advance/focus    0.51      0.753     0.694                      0.037                     0.246          0.963         0.745        0.853         0.868   13.922
                advance/nearest  0.53      0.768     0.659                      0.033                     0.284          0.966         0.709        0.877         0.849   12.866
                reach/focus      0.51      0.753     0.694                      0.037                     0.246          0.963         0.745        0.853         0.868   13.922
                kite/focus       0.99      0.820     0.527                      0.032                     0.360          0.968         0.588        0.999         0.601   12.376
                kite/nearest     0.98      0.840     0.501                      0.028                     0.391          0.971         0.557        0.997         0.571   11.181
         kite   advance/focus    0.43      0.747     0.722                      0.039                     0.140          0.960         0.607        0.827         0.902   12.499
                kite/focus       1.00      0.843     0.527                      0.027                     0.302          0.971         0.510        1.000         0.593   11.398
```

Battles per row: 100.

## Read-outs (facts only)

- Kite's firing rule alone (`reach`) plays like advance (S1 65% vs 66%, S2 64% vs 65%, S3 identical): the gain
  comes from the step back, not from shooting only inside own range.
- Against attack-move, kite lowers Red's duty by 0.05-0.17 (S1 0.68 → 0.58, S3 0.69 → 0.53) and raises Blue's by
  0.03-0.07; Red's weapons are ready with no Blue unit in range more often (S1 0.22 → 0.32, S3 0.25 → 0.36).
- Kiting is not special to Blue: a kiting Red against an advancing Blue cuts Blue's wins from 66 to 34% (S1), 65 to
  27% (S2), 51 to 43% (S3).
- With both sides kiting Blue still wins 87% (S1), 86% (S2), 100% (S3). Not explained: Red's kite is set through
  SMAClite commands (attack-move between steps back, nearest target) and is not an exact copy of Blue's.
- Run time: 46 s for the 2,100 battles (8 workers).
