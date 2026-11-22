# The kite variant: fit and checks K1-K3

Plan: `32` (approved 22 Nov, rules fixed before collection).
Script: `experiments/planning/kite_refit.py`.
Fitted on exp08 train + exp11 train (kite in the schedule).

## Equations

```
H_B^marine' = -0.1789·R[marine]*s[marine] -0.1666·R[marauder]*s[marine]
H_B^marauder' = -0.0592·R[marine]*s[marauder] -0.1135·R[marauder]*s[marauder]
H_R^marine' = -0.1923·R[marine]*s[marine] -0.1754·R[marauder]*s[marine]
H_R^marauder' = -0.0615·R[marine]*s[marauder] -0.1182·R[marauder]*s[marauder]
(ln A_B^marine)' = (0.567) · (ln H_B^marine)'
(ln A_B^marauder)' = (0.450) · (ln H_B^marauder)'
(ln A_R^marine)' = (0.511 +0.146·focus) · (ln H_R^marine)'
(ln A_R^marauder)' = (0.373 +0.132·focus) · (ln H_R^marauder)'
d' = -0.344·walk_r*wd +0.697·danger*wd -5.454·close*adv -3.081·close*hold*far -5.252·close*hold*contact -5.472·close*kite
kite gate (v, c, bb, br, eb, er, w, a): 5.472, 7.778, -0.442, 0.264, 0.047, 0.341, 0.975, 0.098
```

Share of training steps by movement: {'advance': 0.4523073091159279, 'hold': 0.27815615347081435, 'withdraw': 0.20398499019013377, 'kite': 0.06555154722312392}; rows 979626.

## Why the kite terms were dropped (diagnostic, after the fit)

Least squares on each equation's kept terms plus their kite terms, no threshold. 'relative to base': kite coefficient / the matching base coefficient (−0.2 = 20% less fire under kite); 'standardised': the effect size STLSQ compares with 0.075.

```
       equation                         term    coef  relative to base  standardised
  H_blue^marine     R[marine]*s[marine]*kite -0.0092            0.0518       -0.0084
  H_blue^marine   R[marauder]*s[marine]*kite -0.0131            0.0790       -0.0109
H_blue^marauder   R[marine]*s[marauder]*kite  0.0034           -0.0565        0.0068
H_blue^marauder R[marauder]*s[marauder]*kite -0.0080            0.0710       -0.0125
   H_red^marine     R[marine]*s[marine]*kite -0.0208            0.1100       -0.0223
   H_red^marine   R[marauder]*s[marine]*kite -0.1299            0.7563       -0.0633
 H_red^marauder   R[marine]*s[marauder]*kite -0.0188            0.3133       -0.0362
 H_red^marauder R[marauder]*s[marauder]*kite -0.0311            0.2659       -0.0387
```

## K1: exp08 val, kite variant vs frozen (error at the horizon)

```
  h       error  frozen   kite  ratio  difference                95%  passes (ratio <= 1.05)
1.0 team health  0.0401 0.0401 0.9998     -0.0000 [-0.0000, +0.0000]                    True
1.0    distance  0.3014 0.3019 1.0017      0.0005 [+0.0002, +0.0008]                    True
2.0 team health  0.0691 0.0695 1.0051      0.0004 [+0.0003, +0.0004]                    True
2.0    distance  0.4318 0.4333 1.0035      0.0015 [+0.0010, +0.0020]                    True
4.0 team health  0.1281 0.1292 1.0091      0.0012 [+0.0010, +0.0013]                    True
4.0    distance  0.5986 0.6015 1.0049      0.0029 [+0.0021, +0.0038]                    True
```

## K2: exp11 val, kite and advance windows

```
  h  model  window  starts  team health error  distance error
1.0   kite advance    1805             0.0494          0.1679
1.0   kite    kite    1459             0.0584          0.2436
1.0 frozen advance    1805             0.0494          0.1678
1.0 frozen    kite    1459             0.0661          0.3296
2.0   kite advance    1143             0.0747          0.2149
2.0   kite    kite     889             0.0866          0.3076
2.0 frozen advance    1143             0.0741          0.2144
2.0 frozen    kite     889             0.1146          0.4607
4.0   kite advance     426             0.1359          0.2478
4.0   kite    kite     275             0.1400          0.4501
4.0 frozen advance     426             0.1337          0.2483
4.0 frozen    kite     275             0.2239          0.7077
```

Kite / advance team health error at 2 s (kite variant): 1.159; passes (<= 1.5): True.

Reach closure on the steps:

```
movement side  measured  closure  mean abs error
 advance blue    0.4176   0.4121          0.0622
 advance  red    0.3247   0.3139          0.0437
    kite blue    0.3223   0.3795          0.0876
    kite  red    0.2786   0.2638          0.0641
```

## K3: counterfactual sets, seven candidates (exp08 cf seeds)

States: 598; largest state difference at the branch 0.0e+00.

```
  h                                 ranker  regret  picks kite  picks the true best  true best is kite
1.0                                 frozen  0.0281      0.3796               0.2023                NaN
1.0                                   kite  0.0283      0.4130               0.2074                NaN
1.0                 always advance_nearest  0.0171      0.0000               0.3411                NaN
1.0                   always advance_focus  0.0298      0.0000               0.1789                NaN
1.0                      always kite_focus  0.0269      1.0000               0.2542                NaN
1.0 best fixed in hindsight (kite_nearest)  0.0136      1.0000               0.4716             0.5117
2.0                                 frozen  0.0516      0.3629               0.1622                NaN
2.0                                   kite  0.0520      0.4264               0.1622                NaN
2.0                 always advance_nearest  0.0364      0.0000               0.2776                NaN
2.0                   always advance_focus  0.0545      0.0000               0.1304                NaN
2.0                      always kite_focus  0.0461      1.0000               0.2057                NaN
2.0 best fixed in hindsight (kite_nearest)  0.0195      1.0000               0.4615             0.5870
4.0                                 frozen  0.0887      0.3161               0.1773                NaN
4.0                                   kite  0.0864      0.4281               0.1806                NaN
4.0                 always advance_nearest  0.0903      0.0000               0.1472                NaN
4.0                   always advance_focus  0.0982      0.0000               0.1538                NaN
4.0                      always kite_focus  0.0672      1.0000               0.2575                NaN
4.0 best fixed in hindsight (kite_nearest)  0.0310      1.0000               0.5017             0.6923
```

```
  h                     pair  difference                95%
1.0 kite − always kite_focus      0.0014 [-0.0005, +0.0034]
1.0            kite − frozen      0.0001 [-0.0008, +0.0010]
2.0 kite − always kite_focus      0.0059 [+0.0029, +0.0093]
2.0            kite − frozen      0.0004 [-0.0012, +0.0021]
4.0 kite − always kite_focus      0.0191 [+0.0129, +0.0258]
4.0            kite − frozen     -0.0023 [-0.0068, +0.0026]
```

K3 passes: False.

## Read-outs against the plan (`32`)

| step | rule | result |
|---|---|---|
| fit | kite terms in health kept at 0.075? | **none kept**; only close·kite in d' (−5.47, as advance −5.45). Kite was 6.6% of training steps |
| K1 | errors within +5% of frozen | passes (ratios 1.000-1.009) |
| K2 | kite / advance error at 2 s ≤ 1.5 | passes (1.16); the frozen model's ratio would be 1.55 |
| K3 | regret below always kite/focus at 2 s or 4 s | **fails**: kite variant 0.052 vs 0.046 (2 s), 0.086 vs 0.067 (4 s); same as frozen |
| K4 | closed loop only if K1-K3 pass | **not run** |

Facts from the diagnostic: under kite, Red's fire per Red unit in reach is about the same (Blue's health
equations: kite coefficients +5 to +8% of the base, standardised ≤ 0.013), while Blue's fire per Blue unit in reach is
higher (Red's health equations: +11 to +76%, standardised 0.022-0.063, just below 0.075). The reach closure
overpredicts Blue units in reach on kite steps (0.38 vs measured 0.32). On the counterfactual sets, a kite
candidate is the true best in 51-69% of states.
