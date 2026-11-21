# Health compartments: the flows (relative wounded, train only)

Script: `experiments/planning/compartments.py flows`. A unit is wounded once below half of its own start
health. Windows of 1 s every 0.5 s, steady Blue intent, in contact, not withdrawing; CV R² over 5 folds
by battle. d = measured damage on the side's type in the window (full units); d-hat = the frozen health
equation's rate at the window's start × 1 s (what a rollout would have). w = W0 / A0 (wounded share).
Exploratory: no model is changed and nothing is tested on val or cf.
**Only sides that started with at least 80% of full health.**

## S1

CV R² per flow and predictor set:

```
side                                                    blue        red
type                                                  marine     marine
windows                                            20388.000  20663.000
healthy → wounded | d, no state                        0.159      0.198
healthy → wounded | d, compartments                    0.183      0.204
wounded → dead | d, no state                           0.005      0.001
wounded → dead | d, compartments                       0.471      0.586
healthy → dead | d, no state                           0.232      0.352
healthy → dead | d, compartments                       0.284      0.407
all kills | d, no state                                0.124      0.207
all kills | d, compartments                            0.369      0.455
healthy → wounded | d-hat, no state                    0.065      0.090
healthy → wounded | d-hat, compartments                0.116      0.109
wounded → dead | d-hat, no state                       0.058      0.055
wounded → dead | d-hat, compartments                   0.530      0.627
healthy → dead | d-hat, no state                       0.220      0.276
healthy → dead | d-hat, compartments                   0.309      0.387
all kills | d-hat, no state                            0.252      0.316
all kills | d-hat, compartments                        0.457      0.494
wounded → dead | where damage lands (per-unit)         0.425      0.526
healthy → wounded | where damage lands (per-unit)      0.204      0.217
all kills | d, power law                               0.212      0.350
```

Fitted flows (compartment set, d-hat), at the median wounded share:

```
                               median w  per unit of d-hat, nearest   focus  per wounded unit, nearest  focus 
side type   flow                                                                                              
blue marine healthy → wounded    0.1667                      0.3808  0.3545                    -0.1234 -0.1227
            wounded → dead       0.1667                      0.6469  0.6604                     0.2343  0.2095
red  marine healthy → wounded    0.1250                      0.4512  0.1367                    -0.0136 -0.1112
            wounded → dead       0.1250                      0.3828  0.3940                     0.1988  0.4283
```

## S2

CV R² per flow and predictor set:

```
side                                                    blue                   red           
type                                                marauder     marine   marauder     marine
windows                                            28222.000  26561.000  30190.000  24491.000
healthy → wounded | d, no state                        0.217      0.244      0.248      0.240
healthy → wounded | d, compartments                    0.287      0.283      0.317      0.262
wounded → dead | d, no state                           0.020      0.023      0.046      0.023
wounded → dead | d, compartments                       0.227      0.401      0.313      0.569
healthy → dead | d, no state                           0.084      0.207      0.055      0.323
healthy → dead | d, compartments                       0.107      0.253      0.074      0.383
all kills | d, no state                                0.042      0.114      0.069      0.231
all kills | d, compartments                            0.220      0.361      0.299      0.483
healthy → wounded | d-hat, no state                    0.108      0.095      0.123      0.076
healthy → wounded | d-hat, compartments                0.218      0.171      0.225      0.141
wounded → dead | d-hat, no state                       0.144      0.139      0.185      0.149
wounded → dead | d-hat, compartments                   0.333      0.491      0.401      0.613
healthy → dead | d-hat, no state                       0.057      0.159      0.035      0.216
healthy → dead | d-hat, compartments                   0.091      0.239      0.062      0.354
all kills | d-hat, no state                            0.182      0.271      0.216      0.385
all kills | d-hat, compartments                        0.331      0.464      0.390      0.533
wounded → dead | where damage lands (per-unit)         0.210      0.376      0.287      0.511
healthy → wounded | where damage lands (per-unit)      0.328      0.320      0.361      0.282
all kills | d, power law                               0.162      0.258      0.197      0.420
```

Fitted flows (compartment set, d-hat), at the median wounded share:

```
                                 median w  per unit of d-hat, nearest   focus  per wounded unit, nearest  focus 
side type     flow                                                                                              
blue marauder healthy → wounded    0.0000                      1.3636  1.4428                    -0.0467 -0.0535
              wounded → dead       0.0000                      0.4251  0.4406                     0.0911  0.1098
     marine   healthy → wounded    0.1667                      0.5828  0.6494                    -0.0710 -0.0943
              wounded → dead       0.1667                      0.5730  0.5512                     0.1819  0.2160
red  marauder healthy → wounded    0.0000                      1.4748  1.3906                    -0.0459 -0.0780
              wounded → dead       0.0000                      0.3510  0.4533                     0.1072  0.1571
     marine   healthy → wounded    0.1667                      0.5586  0.3403                    -0.0742 -0.0585
              wounded → dead       0.1667                      0.4154  0.3454                     0.2471  0.5015
```

