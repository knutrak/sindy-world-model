# Residuals of the frozen model against the intents (train sets)

Script: `experiments/planning/intent_residuals.py` (set-up in its docstring).

## S1 (marines model, exp06_intents_typed/train)

```
                          n  residual sd  intent R²  max |cell| / sd
quantity                                                            
health blue           87142       0.0229     0.0359           0.7866
health red            87142       0.0251     0.0419           0.8048
alive blue            87142       0.0587     0.0115           0.3379
alive red             87142       0.0550     0.0112           0.4541
distance              87142       0.4700     0.0864           1.2352
reach blue (closure)  87142       0.0788     0.0368           0.5996
reach red (closure)   87142       0.1146     0.0019           0.1077
depth blue (held)     87142       0.2927     0.0475           0.5508
depth red (held)      87142       0.1185     0.1459           1.6276
width blue (held)     87142       0.4693     0.0828           0.5795
width red (held)      87142       0.2122     0.0518           0.7074
```

Cell means:

```
                      advance/focus contact  advance/focus far  advance/nearest contact  advance/nearest far  hold/focus contact  hold/focus far  hold/nearest contact  hold/nearest far  withdraw contact  withdraw far
health blue                         -0.0026            -0.0180                  -0.0024              -0.0162             -0.0014         -0.0086               -0.0014           -0.0082            0.0040       -0.0040
health red                           0.0028            -0.0186                  -0.0034              -0.0202              0.0041         -0.0099               -0.0013           -0.0101            0.0000        0.0000
alive blue                          -0.0078             0.0168                  -0.0010               0.0198             -0.0059          0.0084               -0.0011            0.0090            0.0046        0.0018
alive red                           -0.0053             0.0184                  -0.0034               0.0250             -0.0033          0.0079               -0.0016            0.0105            0.0000        0.0000
distance                             0.0304             0.0172                   0.1007               0.0556              0.0247          0.0733                0.0950            0.0966            0.0357       -0.5805
reach blue (closure)                -0.0101            -0.0027                  -0.0076              -0.0028             -0.0148         -0.0017               -0.0117           -0.0026           -0.0473       -0.0015
reach red (closure)                  0.0009            -0.0037                   0.0007              -0.0046             -0.0048         -0.0046                0.0010           -0.0055           -0.0124       -0.0061
depth blue (held)                   -0.0820            -0.1353                  -0.0884              -0.1612             -0.0633         -0.0815               -0.0750           -0.0805            0.0782       -0.0520
depth red (held)                    -0.0787            -0.1816                  -0.0727              -0.1928             -0.0716         -0.0831               -0.0683           -0.0795            0.0187       -0.0197
width blue (held)                   -0.0894             0.0799                  -0.0774               0.1188             -0.0665          0.0521               -0.0580            0.0573            0.2720        0.0912
width red (held)                    -0.0613             0.1439                  -0.0244               0.1501             -0.0579          0.0556               -0.0279            0.0530           -0.0699       -0.0033
```

## S2 (mixed model, exp08_mixed_typed/train)

```
                           n  residual sd  intent R²  max |cell| / sd
quantity                                                             
health blue           131848       0.0289     0.0271           0.9511
health red            131848       0.0295     0.0585           1.1854
alive blue            131848       0.0534     0.0048           0.1574
alive red             131848       0.0516     0.0219           0.3470
distance              131848       0.4926     0.0450           0.4545
reach blue (closure)  131848       0.0908     0.0550           0.7761
reach red (closure)   131848       0.1263     0.0066           0.2046
depth blue (held)     131848       0.3459     0.0814           0.6268
depth red (held)      131848       0.1411     0.1157           1.4047
width blue (held)     131848       0.5105     0.0888           0.7291
width red (held)      131848       0.2256     0.0444           0.5456
```

Cell means:

```
                      advance/focus contact  advance/focus far  advance/nearest contact  advance/nearest far  hold/focus contact  hold/focus far  hold/nearest contact  hold/nearest far  withdraw contact  withdraw far
health blue                         -0.0048            -0.0275                  -0.0043              -0.0272             -0.0028         -0.0089               -0.0032           -0.0074           -0.0061       -0.0036
health red                           0.0002            -0.0312                  -0.0067              -0.0349              0.0035         -0.0068               -0.0043           -0.0065            0.0000        0.0000
alive blue                          -0.0078             0.0024                  -0.0040               0.0058             -0.0084          0.0025               -0.0048            0.0031           -0.0028        0.0008
alive red                           -0.0179            -0.0020                  -0.0066               0.0033             -0.0169          0.0027               -0.0053            0.0049            0.0000        0.0000
distance                             0.0694            -0.1786                   0.0972              -0.1975              0.0669          0.0301                0.1006            0.0521           -0.0551       -0.2239
reach blue (closure)                -0.0048             0.0022                  -0.0048               0.0005             -0.0140         -0.0010               -0.0144            0.0000           -0.0705       -0.0072
reach red (closure)                  0.0033             0.0156                   0.0041               0.0145             -0.0055          0.0022                0.0015            0.0051           -0.0258        0.0063
depth blue (held)                   -0.0820            -0.1647                  -0.0761              -0.1808             -0.0655         -0.0632               -0.0678           -0.0691            0.2168       -0.0551
depth red (held)                    -0.0705            -0.1982                  -0.0655              -0.1936             -0.0666         -0.0747               -0.0605           -0.0701            0.0428       -0.0103
width blue (held)                   -0.0944             0.0700                  -0.0780               0.1123             -0.0711          0.0385               -0.0594            0.0496            0.3723        0.1229
width red (held)                    -0.0466             0.1231                  -0.0330               0.1184             -0.0489          0.0417               -0.0293            0.0382           -0.1176       -0.0283
```
