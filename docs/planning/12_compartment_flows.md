# Health compartments: the flows (relative wounded, train only)

Script: `experiments/planning/compartments.py flows`. A unit is wounded once below half of its own start
health. Windows of 1 s every 0.5 s, steady Blue intent, in contact, not withdrawing; CV R² over 5 folds
by battle. d = measured damage on the side's type in the window (full units); d-hat = the frozen health
equation's rate at the window's start × 1 s (what a rollout would have). w = W0 / A0 (wounded share).
Exploratory: no model is changed and nothing is tested on val or cf.

## S1

CV R² per flow and predictor set:

```
side                                                    blue        red
type                                                  marine     marine
windows                                            50800.000  50800.000
healthy → wounded | d, no state                        0.147      0.197
healthy → wounded | d, compartments                    0.163      0.203
wounded → dead | d, no state                           0.006      0.001
wounded → dead | d, compartments                       0.526      0.621
healthy → dead | d, no state                           0.265      0.351
healthy → dead | d, compartments                       0.332      0.414
all kills | d, no state                                0.169      0.220
all kills | d, compartments                            0.394      0.448
healthy → wounded | d-hat, no state                    0.047      0.086
healthy → wounded | d-hat, compartments                0.083      0.102
wounded → dead | d-hat, no state                       0.046      0.044
wounded → dead | d-hat, compartments                   0.560      0.650
healthy → dead | d-hat, no state                       0.247      0.257
healthy → dead | d-hat, compartments                   0.353      0.376
all kills | d-hat, no state                            0.273      0.293
all kills | d-hat, compartments                        0.451      0.456
wounded → dead | where damage lands (per-unit)         0.479      0.564
healthy → wounded | where damage lands (per-unit)      0.179      0.212
all kills | d, power law                               0.319      0.410
```

Fitted flows (compartment set, d-hat), at the median wounded share:

```
                               median w  per unit of d-hat, nearest   focus  per wounded unit, nearest  focus 
side type   flow                                                                                              
blue marine healthy → wounded    0.1429                      0.3805  0.3126                    -0.1028 -0.1079
            wounded → dead       0.1429                      0.5789  0.5951                     0.3132  0.2788
red  marine healthy → wounded    0.1250                      0.4276  0.0879                     0.0147 -0.1158
            wounded → dead       0.1250                      0.3630  0.3670                     0.2505  0.4862
```

## S2

CV R² per flow and predictor set:

```
side                                                    blue                   red           
type                                                marauder     marine   marauder     marine
windows                                            72533.000  68636.000  74876.000  61224.000
healthy → wounded | d, no state                        0.223      0.246      0.253      0.231
healthy → wounded | d, compartments                    0.284      0.276      0.313      0.247
wounded → dead | d, no state                           0.019      0.023      0.040      0.016
wounded → dead | d, compartments                       0.275      0.445      0.348      0.591
healthy → dead | d, no state                           0.101      0.236      0.077      0.347
healthy → dead | d, compartments                       0.129      0.288      0.103      0.407
all kills | d, no state                                0.052      0.149      0.077      0.253
all kills | d, compartments                            0.259      0.384      0.318      0.484
healthy → wounded | d-hat, no state                    0.107      0.084      0.113      0.063
healthy → wounded | d-hat, compartments                0.211      0.145      0.209      0.116
wounded → dead | d-hat, no state                       0.149      0.132      0.177      0.136
wounded → dead | d-hat, compartments                   0.372      0.517      0.426      0.627
healthy → dead | d-hat, no state                       0.064      0.187      0.045      0.220
healthy → dead | d-hat, compartments                   0.105      0.278      0.083      0.364
all kills | d-hat, no state                            0.201      0.301      0.222      0.385
all kills | d-hat, compartments                        0.361      0.470      0.398      0.512
wounded → dead | where damage lands (per-unit)         0.249      0.411      0.315      0.520
healthy → wounded | where damage lands (per-unit)      0.323      0.308      0.357      0.264
all kills | d, power law                               0.168      0.302      0.212      0.462
```

Fitted flows (compartment set, d-hat), at the median wounded share:

```
                                 median w  per unit of d-hat, nearest   focus  per wounded unit, nearest  focus 
side type     flow                                                                                              
blue marauder healthy → wounded    0.0000                      1.4871  1.5043                    -0.0564 -0.0615
              wounded → dead       0.0000                      0.4636  0.4692                     0.1322  0.1458
     marine   healthy → wounded    0.1429                      0.6250  0.6092                    -0.0707 -0.0813
              wounded → dead       0.1429                      0.5326  0.5262                     0.2407  0.2550
red  marauder healthy → wounded    0.0000                      1.5363  1.4150                    -0.0415 -0.0899
              wounded → dead       0.0000                      0.3278  0.4392                     0.1216  0.1975
     marine   healthy → wounded    0.1667                      0.5332  0.3121                    -0.0456 -0.0508
              wounded → dead       0.1667                      0.4141  0.3313                     0.2731  0.5418
```

