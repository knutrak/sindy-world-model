# Formation and the S3 mechanism (exploratory diagnostics, 22 Nov 2026)

Scripts: `experiments/planning/formation_check.py`, `experiments/planning/s3_mechanism.py`.

# 1. Does formation limit the frozen model? (val)

The frozen model's error in the margin change, by Red's formation depth at the start (quartiles), with the
shape held fixed (as the model does) and with the true shape (depth, width per side) fed at every step.

## S1 (marines model, val)

```
                               n  margin error, shape held (frozen)  with the true shape  error removed  bias, frozen (pred − true)
h   start states                                                                                                                   
2.0 all                     7946                             0.0358               0.0351         0.0194                      0.0008
    Q2                      2013                             0.0318               0.0321        -0.0097                      0.0039
    Q3                      2141                             0.0386               0.0376         0.0259                      0.0024
    Q4 (strung out)         2202                             0.0459               0.0432         0.0573                     -0.0039
    Red depth Q1 (compact)  1590                             0.0232               0.0243        -0.0484                      0.0013
4.0 all                     5947                             0.0745               0.0707         0.0509                      0.0001
    Q2                      1361                             0.0607               0.0615        -0.0141                      0.0024
    Q3                      1744                             0.0761               0.0739         0.0288                      0.0046
    Q4 (strung out)         2065                             0.0953               0.0854         0.1043                     -0.0054
    Red depth Q1 (compact)   777                             0.0400               0.0408        -0.0194                      0.0003
```

## S2 (mixed model, val)

```
                                n  margin error, shape held (frozen)  with the true shape  error removed  bias, frozen (pred − true)
h   start states                                                                                                                    
2.0 all                     13520                             0.0440               0.0432         0.0177                      0.0008
    Q2                       3470                             0.0408               0.0405         0.0060                      0.0035
    Q3                       3552                             0.0445               0.0439         0.0135                      0.0039
    Q4 (strung out)          3603                             0.0566               0.0544         0.0392                     -0.0019
    Red depth Q1 (compact)   2895                             0.0317               0.0318        -0.0043                     -0.0031
4.0 all                     11520                             0.0819               0.0803         0.0193                     -0.0010
    Q2                       2940                             0.0693               0.0715        -0.0321                      0.0066
    Q3                       3303                             0.0809               0.0810        -0.0015                      0.0035
    Q4 (strung out)          3516                             0.1098               0.1023         0.0684                     -0.0100
    Red depth Q1 (compact)   1761                             0.0492               0.0499        -0.0147                     -0.0040
```

# 2. Why the model values an early withdrawal in S3 but not S1

A small counterfactual set (150 seeds per scenario, seeds 19 800 000 + / 19 800 500 +): random intents up
to an early branch (0.5-4 s), then advance/focus, hold/focus or withdraw for 2 s, then advance/nearest; the
frozen model rolled out from the branch state under the same controls. Withdraw minus the other branch, mean
over states, over 3 s.

## S3: withdraw for 2 s at an early branch (0.5-4 s), effect over 3 s
(differences of means over states; losses in ten units of the first type)

```
withdraw minus              advance/focus  hold/focus
margin change, true                0.0191      0.0183
margin change, model               0.0184      0.0161
blue loss, true                   -0.1180     -0.1168
blue loss, model                  -0.1874     -0.1793
red loss, true                    -0.0989     -0.0985
red loss, model                   -0.1690     -0.1632
sign agreement                     0.5800      0.5733
model says withdraw better         0.7467      0.7267
truly better                       0.6067      0.5933
states                           150.0000    150.0000
```

## S1: withdraw for 2 s at an early branch (0.5-4 s), effect over 3 s
(differences of means over states; losses in ten units of the first type)

```
withdraw minus              advance/focus  hold/focus
margin change, true               -0.0868     -0.0885
margin change, model              -0.1764     -0.1751
blue loss, true                   -0.0990     -0.0963
blue loss, model                  -0.0876     -0.0839
red loss, true                    -0.1858     -0.1847
red loss, model                   -0.2640     -0.2591
sign agreement                     0.8222      0.8370
model says withdraw better         0.0074      0.0074
truly better                       0.1852      0.1704
states                           135.0000    135.0000
```

