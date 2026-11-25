# Continuous or discrete time

Exploratory (val).

```
Continuous or discrete time? (exploratory, 25 Nov 2026; fit on train, score on val, marines and mixed)

C    the frozen pipeline models: SG-smoothed derivatives, Euler steps of 0.125 s
Dm   the same candidates and closures fitted on raw differences (x[k+m] - x[k]) / (m dt) with the regressors at k, only
     windows with one intent; stepped with m dt (m = 1, 4, 8: 0.125, 0.5, 1 s). Health and distance cutoffs equal C's
     in absolute size (STLSQ's threshold is relative to the target's spread, and raw differences have more spread).
D1t  D1 with the original relative thresholds
Cm   C stepped with m dt
Rule, fixed before the run: discrete time is better if Dm's health error at 2 s is below C's (paired 95% interval below 0,
bootstrap over battles) in both scenarios.
```

## marines

```
model                          C                  C4                  C8                  D1                 D1t                  D4                  D8
step s                     0.125                 0.5                 1.0               0.125               0.125                 0.5                 1.0
health 1s                 0.0164              0.0168              0.0195              0.0162              0.0162              0.0173              0.0199
margin R2 1s              0.4308              0.4219              0.3577              0.4262              0.4262              0.4358              0.4471
- C, 1s 95%   [+0.0000, +0.0000]  [+0.0004, +0.0005]  [+0.0029, +0.0034]  [-0.0002, -0.0001]  [-0.0002, -0.0001]  [+0.0008, +0.0011]  [+0.0032, +0.0040]
health 2s                 0.0277              0.0281                0.03              0.0272              0.0272              0.0283              0.0315
margin R2 2s              0.4012              0.4169              0.3889              0.3941              0.3941              0.4568              0.4891
alive 2s                  0.5505              0.5473              0.5523              0.5479              0.5479              0.5353               0.616
distance 2s               0.3594              0.3686               0.439               0.349               0.349              0.3515               0.368
- C, 2s 95%   [+0.0000, +0.0000]  [+0.0002, +0.0005]  [+0.0019, +0.0026]  [-0.0006, -0.0004]  [-0.0006, -0.0004]  [+0.0003, +0.0009]  [+0.0031, +0.0046]
health 4s                 0.0523              0.0524              0.0553              0.0513              0.0513              0.0517              0.0544
margin R2 4s              0.0798              0.1212              0.1028              0.0686              0.0686              0.2059              0.2934
- C, 4s 95%   [+0.0000, +0.0000]  [-0.0001, +0.0003]  [+0.0024, +0.0037]  [-0.0014, -0.0005]  [-0.0014, -0.0005]  [-0.0014, +0.0002]  [+0.0008, +0.0034]
```

C:

```
H_B^marine' = -0.1770·R[marine]*s[marine]
H_R^marine' = -0.1861·R[marine]*s[marine]
(ln A_B^marine)' = (0.662) · (ln H_B^marine)'
(ln A_R^marine)' = (0.562 +0.260·focus) · (ln H_R^marine)'
d' = -0.485·walk_r*wd +0.575·danger*wd -6.008·close*adv -3.183·close*hold*far -5.915·close*hold*contact
```

D1:

```
H_B^marine' = -0.1842·R[marine]*s[marine]
H_R^marine' = -0.1922·R[marine]*s[marine]
(ln A_B^marine)' = (0.729) · (ln H_B^marine)'
(ln A_R^marine)' = (0.633 +0.280·focus) · (ln H_R^marine)'
d' = -0.638·walk_r*wd +0.774·danger*wd -5.576·close*adv -3.182·close*hold*far -5.621·close*hold*contact
```

D1t:

```
H_B^marine' = -0.1842·R[marine]*s[marine]
H_R^marine' = -0.1922·R[marine]*s[marine]
(ln A_B^marine)' = (0.729) · (ln H_B^marine)'
(ln A_R^marine)' = (0.633 +0.280·focus) · (ln H_R^marine)'
d' = -0.638·walk_r*wd +0.774·danger*wd -5.576·close*adv -3.182·close*hold*far -5.621·close*hold*contact
```

D4:

```
H_B^marine' = -0.1688·R[marine]*s[marine]
H_R^marine' = -0.1799·R[marine]*s[marine]
(ln A_B^marine)' = (0.759) · (ln H_B^marine)'
(ln A_R^marine)' = (0.675 +0.246·focus) · (ln H_R^marine)'
d' = -0.585·walk_r*wd +0.647·danger*wd -5.626·close*adv -3.264·close*hold*far -5.882·close*hold*contact
```

D8:

```
H_B^marine' = -0.1162·R[marine]*s[marine] -0.0323·A[marine]
H_R^marine' = -0.1345·R[marine]*s[marine] -0.0313·A[marine]
(ln A_B^marine)' = (0.745) · (ln H_B^marine)'
(ln A_R^marine)' = (0.682 +0.219·focus) · (ln H_R^marine)'
d' = -0.534·walk_r*wd +0.554·danger*wd -5.305·close*adv -3.205·close*hold*far -5.527·close*hold*contact
```

## mixed

```
model                          C                  C4                  C8                  D1                 D1t                  D4                  D8
step s                     0.125                 0.5                 1.0               0.125               0.125                 0.5                 1.0
health 1s                   0.02              0.0202              0.0215              0.0198              0.0205              0.0207              0.0225
margin R2 1s              0.4029              0.4189              0.3992              0.3996               0.334              0.4271               0.419
- C, 1s 95%   [+0.0000, +0.0000]  [+0.0000, +0.0002]  [+0.0012, +0.0017]  [-0.0003, -0.0002]  [+0.0003, +0.0006]  [+0.0006, +0.0008]  [+0.0022, +0.0027]
health 2s                 0.0346              0.0343              0.0346              0.0335              0.0351              0.0346              0.0369
margin R2 2s              0.3743              0.4154              0.4355              0.3639              0.2808              0.4424              0.4689
alive 2s                  0.5014              0.4969              0.4935              0.4863               0.483              0.4835              0.5173
distance 2s               0.4318              0.4372              0.4887              0.4082              0.4082              0.4108              0.4207
- C, 2s 95%   [+0.0000, +0.0000]  [-0.0004, -0.0002]  [-0.0004, +0.0003]  [-0.0013, -0.0009]  [+0.0002, +0.0008]  [-0.0002, +0.0003]  [+0.0018, +0.0028]
health 4s                  0.064              0.0635              0.0646               0.061              0.0643              0.0623              0.0641
margin R2 4s              0.1476              0.1988              0.2213              0.1303              0.0343              0.2649              0.3386
- C, 4s 95%   [+0.0000, +0.0000]  [-0.0007, -0.0003]  [+0.0001, +0.0011]  [-0.0036, -0.0023]  [-0.0006, +0.0012]  [-0.0024, -0.0010]  [-0.0010, +0.0010]
```

C:

```
H_B^marine' = -0.1784·R[marine]*s[marine] -0.1651·R[marauder]*s[marine]
H_B^marauder' = -0.0600·R[marine]*s[marauder] -0.1127·R[marauder]*s[marauder]
H_R^marine' = -0.1892·R[marine]*s[marine] -0.1719·R[marauder]*s[marine]
H_R^marauder' = -0.0599·R[marine]*s[marauder] -0.1170·R[marauder]*s[marauder]
(ln A_B^marine)' = (0.578) · (ln H_B^marine)'
(ln A_B^marauder)' = (0.444) · (ln H_B^marauder)'
(ln A_R^marine)' = (0.506 +0.155·focus) · (ln H_R^marine)'
(ln A_R^marauder)' = (0.376 +0.139·focus) · (ln H_R^marauder)'
d' = -0.340·walk_r*wd +0.708·danger*wd -5.434·close*adv -3.081·close*hold*far -5.263·close*hold*contact
```

D1:

```
H_B^marine' = -0.1899·R[marine]*s[marine] -0.1675·R[marauder]*s[marine]
H_B^marauder' = -0.0619·R[marine]*s[marauder] -0.1142·R[marauder]*s[marauder]
H_R^marine' = -0.1942·R[marine]*s[marine] -0.1848·R[marauder]*s[marine]
H_R^marauder' = -0.0599·R[marine]*s[marauder] -0.1232·R[marauder]*s[marauder]
(ln A_B^marine)' = (0.725) · (ln H_B^marine)'
(ln A_B^marauder)' = (0.643) · (ln H_B^marauder)'
(ln A_R^marine)' = (0.654 +0.184·focus) · (ln H_R^marine)'
(ln A_R^marauder)' = (0.529 +0.209·focus) · (ln H_R^marauder)'
d' = -0.512·walk_r*wd +0.924·danger*wd -4.010·close*adv -3.025·close*hold*far -4.202·close*hold*contact
```

D1t:

```
H_B^marine' = -0.1899·R[marine]*s[marine] -0.1675·R[marauder]*s[marine]
H_B^marauder' = -0.1486·R[marauder]*s[marauder]
H_R^marine' = -0.1942·R[marine]*s[marine] -0.1848·R[marauder]*s[marine]
H_R^marauder' = -0.0599·R[marine]*s[marauder] -0.1232·R[marauder]*s[marauder]
(ln A_B^marine)' = (0.725) · (ln H_B^marine)'
(ln A_B^marauder)' = (0.624) · (ln H_B^marauder)'
(ln A_R^marine)' = (0.654 +0.184·focus) · (ln H_R^marine)'
(ln A_R^marauder)' = (0.529 +0.209·focus) · (ln H_R^marauder)'
d' = -0.512·walk_r*wd +0.924·danger*wd -4.010·close*adv -3.025·close*hold*far -4.202·close*hold*contact
```

D4:

```
H_B^marine' = -0.1647·R[marine]*s[marine] -0.1559·R[marauder]*s[marine]
H_B^marauder' = -0.0580·R[marine]*s[marauder] -0.1055·R[marauder]*s[marauder]
H_R^marine' = -0.1766·R[marine]*s[marine] -0.1582·R[marauder]*s[marine]
H_R^marauder' = -0.0605·R[marine]*s[marauder] -0.1101·R[marauder]*s[marauder]
(ln A_B^marine)' = (0.769) · (ln H_B^marine)'
(ln A_B^marauder)' = (0.673) · (ln H_B^marauder)'
(ln A_R^marine)' = (0.708 +0.163·focus) · (ln H_R^marine)'
(ln A_R^marauder)' = (0.561 +0.184·focus) · (ln H_R^marauder)'
d' = -0.571·walk_r*wd +0.880·danger*wd -5.245·close*adv -3.245·close*hold*far -5.256·close*hold*contact
```

D8:

```
H_B^marine' = -0.1112·R[marine]*s[marine] -0.0245·A[marine] -0.1317·R[marauder]*s[marine]
H_B^marauder' = -0.0533·R[marine]*s[marauder] -0.1030·R[marauder]*s[marauder]
H_R^marine' = -0.1267·R[marine]*s[marine] -0.0270·A[marine] -0.1272·R[marauder]*s[marine]
H_R^marauder' = -0.0565·R[marine]*s[marauder] -0.1093·R[marauder]*s[marauder]
(ln A_B^marine)' = (0.773) · (ln H_B^marine)'
(ln A_B^marauder)' = (0.690) · (ln H_B^marauder)'
(ln A_R^marine)' = (0.704 +0.168·focus) · (ln H_R^marine)'
(ln A_R^marauder)' = (0.587 +0.161·focus) · (ln H_R^marauder)'
d' = -0.509·walk_r*wd +0.707·danger*wd -5.130·close*adv -3.253·close*hold*far -5.231·close*hold*contact
```

## Rule (health error at 2 s below C's in both scenarios)

- D1: marines True, mixed True
- D1t: marines True, mixed False
- D4: marines False, mixed False
- D8: marines False, mixed False
- C4: marines False, mixed True
- C8: marines False, mixed False
