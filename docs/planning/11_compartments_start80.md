# Health compartments: a train-only check

Script: `experiments/planning/compartments.py` (set-up in its docstring). Train sets only, windows of 1 s
every 0.5 s with a steady Blue intent, both sides alive. Wounded: below half health. Damage in full units.
Exploratory, like `05_intent_residuals.md`: no model is changed and nothing is tested on val or cf.
**Only sides that started with at least 80% of full health.**

## S1

### 1. Where kills come from

```
                         windows  kills per window  share of kills from units wounded at the start  share of alive units wounded
side type   Blue target                                                                                                         
blue marine nearest        11428             0.644                                           0.808                         0.174
            focus          11722             0.646                                           0.793                         0.178
red  marine nearest        11722             0.757                                           0.736                         0.189
            focus          11509             1.020                                           0.598                         0.135
```

### 2. Kill laws in contact (CV R² of kills in the next 1 s)

```
             power law (A0, H loss, focus)  compartment: damage × wounded share, W0 (model can know)  where damage lands, linear (needs per-unit fire)  both: power law + wounded share  windows
side type                                                                                                                                                                                       
blue marine                         0.2124                                                    0.3262                                            0.2951                           0.2481    20388
red  marine                         0.3498                                                    0.4413                                            0.3833                           0.4018    20663
```

### 3. Focus and the wounded compartment (in contact; focus − nearest)

```
                                     focus − nearest [95%, bootstrap by battle]
side type   quantity                                                           
blue marine wounded at the start                     +0.0083 [-0.0296, +0.0454]
            healthy → wounded in 1 s                 -0.0526 [-0.0724, -0.0326]
            wounded → dead in 1 s                    -0.0081 [-0.0268, +0.0105]
            kills in 1 s                             +0.0028 [-0.0188, +0.0230]
red  marine wounded at the start                     -0.4945 [-0.5332, -0.4567]
            healthy → wounded in 1 s                 -0.3560 [-0.3765, -0.3353]
            wounded → dead in 1 s                    +0.0570 [+0.0361, +0.0771]
            kills in 1 s                             +0.2824 [+0.2556, +0.3074]
```

### 4. Memory: the previous second's focus, without and with the wounded count

```
             previous focus × damage, without W0  with W0  current focus × damage, without W0  with W0   windows
side type                                                                                                       
blue marine                               0.0322   0.0767                              0.0314    0.0153    19022
red  marine                              -0.2513  -0.0663                              0.4956    0.6052    19316
```

## S2

### 1. Where kills come from

```
                           windows  kills per window  share of kills from units wounded at the start  share of alive units wounded
side type     Blue target                                                                                                         
blue marauder nearest        16413             0.138                                           0.959                         0.221
              focus          16284             0.132                                           0.966                         0.219
     marine   nearest        15500             0.336                                           0.881                         0.206
              focus          15332             0.343                                           0.874                         0.201
red  marauder nearest        17440             0.172                                           0.951                         0.235
              focus          17486             0.180                                           0.971                         0.205
     marine   nearest        14459             0.407                                           0.805                         0.215
              focus          14147             0.653                                           0.717                         0.188
```

### 2. Kill laws in contact (CV R² of kills in the next 1 s)

```
               power law (A0, H loss, focus)  compartment: damage × wounded share, W0 (model can know)  where damage lands, linear (needs per-unit fire)  both: power law + wounded share  windows
side type                                                                                                                                                                                         
blue marauder                         0.1621                                                    0.1869                                            0.1534                           0.1615    28222
     marine                           0.2579                                                    0.3138                                            0.2795                           0.2746    26561
red  marauder                         0.1968                                                    0.2554                                            0.2196                           0.2049    30190
     marine                           0.4198                                                    0.4641                                            0.4015                           0.4532    24491
```

### 3. Focus and the wounded compartment (in contact; focus − nearest)

```
                                       focus − nearest [95%, bootstrap by battle]
side type     quantity                                                           
blue marauder wounded at the start                     -0.0147 [-0.0373, +0.0092]
              healthy → wounded in 1 s                 -0.0076 [-0.0168, +0.0008]
              wounded → dead in 1 s                    -0.0044 [-0.0126, +0.0040]
              kills in 1 s                             -0.0058 [-0.0140, +0.0022]
     marine   wounded at the start                     -0.0025 [-0.0315, +0.0259]
              healthy → wounded in 1 s                 -0.0006 [-0.0142, +0.0123]
              wounded → dead in 1 s                    +0.0039 [-0.0087, +0.0167]
              kills in 1 s                             +0.0050 [-0.0091, +0.0187]
red  marauder wounded at the start                     -0.1027 [-0.1281, -0.0781]
              healthy → wounded in 1 s                 -0.0893 [-0.0985, -0.0797]
              wounded → dead in 1 s                    +0.0131 [+0.0037, +0.0216]
              kills in 1 s                             +0.0097 [+0.0006, +0.0191]
     marine   wounded at the start                     -0.1353 [-0.1632, -0.1074]
              healthy → wounded in 1 s                 +0.0069 [-0.0067, +0.0206]
              wounded → dead in 1 s                    +0.1655 [+0.1512, +0.1799]
              kills in 1 s                             +0.2820 [+0.2641, +0.3020]
```

### 4. Memory: the previous second's focus, without and with the wounded count

```
               previous focus × damage, without W0  with W0  current focus × damage, without W0  with W0   windows
side type                                                                                                         
blue marauder                               0.0179   0.0292                              0.0055   -0.0080    26819
     marine                                 0.0773   0.0737                             -0.0244   -0.0080    25158
red  marauder                              -0.1705  -0.0685                              0.3091    0.2715    28865
     marine                                -0.1671  -0.0623                              0.3819    0.4391    23166
```

