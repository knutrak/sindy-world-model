# Health compartments: a train-only check

Script: `experiments/planning/compartments.py` (set-up in its docstring). Train sets only, windows of 1 s
every 0.5 s with a steady Blue intent, both sides alive. Wounded: below half health. Damage in full units.
Exploratory, like `05_intent_residuals.md`: no model is changed and nothing is tested on val or cf.

## S1

### 1. Where kills come from

```
                         windows  kills per window  share of kills from units wounded at the start  share of alive units wounded
side type   Blue target                                                                                                         
blue marine nearest        29020             0.796                                           0.820                         0.228
            focus          29221             0.799                                           0.793                         0.224
red  marine nearest        29020             0.865                                           0.744                         0.248
            focus          29221             1.118                                           0.632                         0.177
```

### 2. Kill laws in contact (CV R² of kills in the next 1 s)

```
             power law (A0, H loss, focus)  compartment: damage × wounded share, W0 (model can know)  where damage lands, linear (needs per-unit fire)  both: power law + wounded share  windows
side type                                                                                                                                                                                       
blue marine                         0.3187                                                    0.3077                                            0.3084                           0.3309    50800
red  marine                         0.4100                                                    0.3986                                            0.3806                           0.4339    50800
```

### 3. Focus and the wounded compartment (in contact; focus − nearest)

```
                                     focus − nearest [95%, bootstrap by battle]
side type   quantity                                                           
blue marine wounded at the start                     -0.0404 [-0.0802, +0.0046]
            healthy → wounded in 1 s                 -0.0726 [-0.0879, -0.0573]
            wounded → dead in 1 s                    -0.0192 [-0.0334, -0.0051]
            kills in 1 s                             +0.0043 [-0.0103, +0.0195]
red  marine wounded at the start                     -0.6349 [-0.6747, -0.5969]
            healthy → wounded in 1 s                 -0.3903 [-0.4054, -0.3757]
            wounded → dead in 1 s                    +0.0725 [+0.0579, +0.0866]
            kills in 1 s                             +0.2726 [+0.2559, +0.2905]
```

### 4. Memory: the previous second's focus, without and with the wounded count

```
             previous focus × damage, without W0  with W0  current focus × damage, without W0  with W0   windows
side type                                                                                                       
blue marine                               0.0387   0.0771                              0.0304    0.0390    46816
red  marine                              -0.2631  -0.0979                              0.5060    0.6245    46816
```

## S2

### 1. Where kills come from

```
                           windows  kills per window  share of kills from units wounded at the start  share of alive units wounded
side type     Blue target                                                                                                         
blue marauder nearest        42201             0.166                                           0.960                         0.288
              focus          42173             0.164                                           0.961                         0.280
     marine   nearest        40053             0.400                                           0.891                         0.251
              focus          39879             0.414                                           0.873                         0.244
red  marauder nearest        43697             0.193                                           0.949                         0.314
              focus          43509             0.190                                           0.980                         0.270
     marine   nearest        36513             0.457                                           0.817                         0.269
              focus          35643             0.721                                           0.743                         0.243
```

### 2. Kill laws in contact (CV R² of kills in the next 1 s)

```
               power law (A0, H loss, focus)  compartment: damage × wounded share, W0 (model can know)  where damage lands, linear (needs per-unit fire)  both: power law + wounded share  windows
side type                                                                                                                                                                                         
blue marauder                         0.1677                                                    0.1511                                            0.1235                           0.1681    72533
     marine                           0.3020                                                    0.2889                                            0.2827                           0.3097    68636
red  marauder                         0.2123                                                    0.2095                                            0.1896                           0.2155    74876
     marine                           0.4623                                                    0.4383                                            0.3984                           0.4803    61224
```

### 3. Focus and the wounded compartment (in contact; focus − nearest)

```
                                       focus − nearest [95%, bootstrap by battle]
side type     quantity                                                           
blue marauder wounded at the start                     -0.0191 [-0.0412, +0.0026]
              healthy → wounded in 1 s                 -0.0130 [-0.0194, -0.0069]
              wounded → dead in 1 s                    -0.0016 [-0.0073, +0.0040]
              kills in 1 s                             -0.0018 [-0.0075, +0.0041]
     marine   wounded at the start                     -0.0159 [-0.0457, +0.0126]
              healthy → wounded in 1 s                 -0.0125 [-0.0216, -0.0035]
              wounded → dead in 1 s                    +0.0029 [-0.0066, +0.0121]
              kills in 1 s                             +0.0109 [+0.0010, +0.0206]
red  marauder wounded at the start                     -0.1548 [-0.1767, -0.1341]
              healthy → wounded in 1 s                 -0.0966 [-0.1029, -0.0905]
              wounded → dead in 1 s                    +0.0037 [-0.0025, +0.0095]
              kills in 1 s                             -0.0030 [-0.0091, +0.0032]
     marine   wounded at the start                     -0.1582 [-0.1831, -0.1340]
              healthy → wounded in 1 s                 +0.0059 [-0.0035, +0.0158]
              wounded → dead in 1 s                    +0.1877 [+0.1779, +0.1982]
              kills in 1 s                             +0.2960 [+0.2836, +0.3088]
```

### 4. Memory: the previous second's focus, without and with the wounded count

```
               previous focus × damage, without W0  with W0  current focus × damage, without W0  with W0   windows
side type                                                                                                         
blue marauder                               0.0439   0.0609                              0.0064   -0.0005    68523
     marine                                 0.0647   0.0947                              0.0142    0.0218    64626
red  marauder                              -0.2125  -0.1192                              0.3356    0.3307    70866
     marine                                -0.1735  -0.0799                              0.4003    0.4736    57214
```

