# One model with the scenario parameters as input: results (exploratory, val)

Plan: `00_plan.md`.

## Rules

- small: E1 better in 0 of 5 (needs 3); E2 worst ratio 1.039 (limit 1.10)
- large: E1 better in 1 of 5 (needs 3); E2 worst ratio 1.049 (limit 1.10)
- large 0.05: E1 better in 1 of 5 (needs 3); E2 worst ratio 1.041 (limit 1.10)
- LO small: E3 ratios marauders 0.988, stalkers 1.045 (limit 1.15)
- LO large: E3 ratios marauders 1.534, stalkers 1.914 (limit 1.15)

```
         army       model  health 2s   ratio  health 4s   R2 2s  distance 2s           - own 95%
0     marines         own     0.0277  1.0000     0.0523  0.4012       0.3594                 NaN
1     marines       small     0.0282  1.0185     0.0522  0.4425       0.3568  [+0.0002, +0.0008]
2     marines       large     0.0271  0.9788     0.0485  0.5127       0.3567  [-0.0009, -0.0003]
3     marines  large 0.05     0.0271  0.9785     0.0484  0.5142       0.3567  [-0.0010, -0.0003]
4     marines    LO small     0.0276  0.9961     0.0509  0.4292       0.3600  [-0.0004, +0.0001]
5     marines    LO large     0.0265  0.9564     0.0484  0.4655       0.3600  [-0.0015, -0.0009]
6   marauders         own     0.0203  1.0000     0.0386  0.3436       0.4666                 NaN
7   marauders       small     0.0201  0.9911     0.0368  0.3885       0.4963  [-0.0004, +0.0000]
8   marauders       large     0.0202  0.9945     0.0358  0.4273       0.4961  [-0.0005, +0.0003]
9   marauders  large 0.05     0.0202  0.9944     0.0358  0.4274       0.4961  [-0.0005, +0.0003]
10  marauders    LO small     0.0201  0.9877     0.0364  0.3889       0.5429  [-0.0005, -0.0000]
11  marauders    LO large     0.0312  1.5343     0.0564 -0.0579       0.5437  [+0.0102, +0.0116]
12      mixed         own     0.0346  1.0000     0.0640  0.3743       0.4318                 NaN
13      mixed       small     0.0345  0.9975     0.0625  0.3906       0.4432  [-0.0002, +0.0001]
14      mixed       large     0.0348  1.0074     0.0611  0.4185       0.4433  [-0.0000, +0.0005]
15      mixed  large 0.05     0.0348  1.0073     0.0610  0.4196       0.4433  [-0.0000, +0.0005]
16      mixed    LO small     0.0346  1.0009     0.0628  0.3838       0.4739  [-0.0002, +0.0002]
17      mixed    LO large     0.0449  1.2999     0.0813  0.1385       0.4744  [+0.0097, +0.0111]
18     triple         own     0.0431  1.0000     0.0799  0.4203       0.4475                 NaN
19     triple       small     0.0431  0.9982     0.0769  0.4323       0.4793  [-0.0003, +0.0002]
20     triple       large     0.0432  1.0013     0.0752  0.4427       0.4795  [-0.0004, +0.0005]
21     triple  large 0.05     0.0432  1.0017     0.0753  0.4423       0.4795  [-0.0004, +0.0006]
22     triple    LO small     0.0433  1.0028     0.0773  0.4333       0.5200  [-0.0002, +0.0005]
23     triple    LO large     0.0691  1.6007     0.1241  0.0680       0.5210  [+0.0240, +0.0278]
24   stalkers         own     0.0145  1.0000     0.0242  0.3911       0.4266                 NaN
25   stalkers       small     0.0150  1.0392     0.0259  0.4068       0.5597  [+0.0004, +0.0007]
26   stalkers       large     0.0152  1.0492     0.0244  0.3931       0.5598  [+0.0005, +0.0010]
27   stalkers  large 0.05     0.0151  1.0413     0.0243  0.3917       0.5598  [+0.0004, +0.0008]
28   stalkers    LO small     0.0151  1.0453     0.0263  0.4090       0.6241  [+0.0005, +0.0008]
29   stalkers    LO large     0.0277  1.9139     0.0523 -0.1506       0.6250  [+0.0124, +0.0140]
30         DM         own     0.0145  1.0000     0.0296  0.5297       0.3372                 NaN
31         DM       small     0.0142  0.9827     0.0282  0.5753       0.3466  [-0.0004, -0.0001]
32         DM       large     0.0184  1.2704     0.0354  0.5180       0.3469  [+0.0037, +0.0042]
33         DM  large 0.05     0.0183  1.2639     0.0351  0.5280       0.3469  [+0.0036, +0.0041]
34         DM    LO small     0.0142  0.9783     0.0278  0.5609       0.3568  [-0.0005, -0.0001]
35         DM    LO large     0.0177  1.2200     0.0348  0.4699       0.3571  [+0.0030, +0.0034]
36         DP         own     0.0491  1.0000     0.0895  0.0476       0.4505                 NaN
37         DP       small     0.0508  1.0350     0.0797  0.1034       0.4449  [+0.0001, +0.0035]
38         DP       large     0.0492  1.0032     0.0795  0.2895       0.4448  [-0.0009, +0.0012]
39         DP  large 0.05     0.0492  1.0035     0.0795  0.2900       0.4448  [-0.0009, +0.0013]
40         DP    LO small     0.0494  1.0066     0.0810  0.0704       0.4510  [-0.0010, +0.0017]
41         DP    LO large     0.0467  0.9527     0.0771  0.2213       0.4507  [-0.0035, -0.0010]
42         CM         own     0.0377  1.0000     0.0689  0.2542       0.3986                 NaN
43         CM       small     0.0393  1.0432     0.0689  0.2803       0.3944  [+0.0010, +0.0023]
44         CM       large     0.0379  1.0071     0.0665  0.3923       0.3945  [-0.0005, +0.0010]
45         CM  large 0.05     0.0379  1.0065     0.0665  0.3945       0.3945  [-0.0005, +0.0010]
46         CM    LO small     0.0385  1.0231     0.0696  0.2369       0.3969  [+0.0004, +0.0013]
47         CM    LO large     0.0368  0.9758     0.0674  0.3213       0.3968  [-0.0015, -0.0003]
48         CP         own     0.0196  1.0000     0.0336  0.3734       0.3390                 NaN
49         CP       small     0.0196  1.0045     0.0331  0.4150       0.3513  [-0.0000, +0.0002]
50         CP       large     0.0200  1.0242     0.0320  0.4596       0.3514  [+0.0002, +0.0008]
51         CP  large 0.05     0.0200  1.0253     0.0319  0.4618       0.3514  [+0.0002, +0.0008]
52         CP    LO small     0.0193  0.9878     0.0323  0.4047       0.3602  [-0.0004, -0.0001]
53         CP    LO large     0.0195  0.9974     0.0331  0.3864       0.3603  [-0.0002, +0.0001]
54         HM         own     0.0385  1.0000     0.0686  0.2018       0.4171                 NaN
55         HM       small     0.0413  1.0717     0.0696  0.1851       0.4123  [+0.0022, +0.0033]
56         HM       large     0.0395  1.0256     0.0668  0.3408       0.4120  [+0.0002, +0.0018]
57         HM  large 0.05     0.0395  1.0258     0.0668  0.3417       0.4120  [+0.0002, +0.0018]
58         HM    LO small     0.0402  1.0430     0.0693  0.1612       0.4156  [+0.0012, +0.0021]
59         HM    LO large     0.0380  0.9859     0.0661  0.2822       0.4158  [-0.0012, +0.0002]
60         A1         own     0.0231  1.0000     0.0448  0.4261       0.3653                 NaN
61         A1       small     0.0232  1.0046     0.0441  0.4999       0.3618  [-0.0002, +0.0004]
62         A1       large     0.0231  1.0008     0.0427  0.5474       0.3622  [-0.0003, +0.0004]
63         A1  large 0.05     0.0231  1.0006     0.0426  0.5493       0.3622  [-0.0004, +0.0004]
64         A1    LO small     0.0229  0.9918     0.0436  0.4795       0.3693  [-0.0005, +0.0001]
65         A1    LO large     0.0224  0.9705     0.0431  0.4960       0.3698  [-0.0010, -0.0004]
66         SM         own     0.0256  1.0000     0.0483  0.2978       0.2958                 NaN
67         SM       small     0.0268  1.0490     0.0508  0.3679       0.3163  [+0.0009, +0.0015]
68         SM       large     0.0256  1.0023     0.0463  0.4574       0.3163  [-0.0003, +0.0005]
69         SM  large 0.05     0.0251  0.9835     0.0456  0.4652       0.3163  [-0.0008, -0.0000]
70         SM    LO small     0.0264  1.0329     0.0495  0.3228       0.3066  [+0.0006, +0.0011]
71         SM    LO large     0.0253  0.9902     0.0475  0.3342       0.3066  [-0.0006, +0.0001]
72         SP         own     0.0294  1.0000     0.0558  0.4276       0.4803                 NaN
73         SP       small     0.0293  0.9965     0.0542  0.4357       0.4521  [-0.0006, +0.0004]
74         SP       large     0.0295  1.0023     0.0530  0.5027       0.4524  [-0.0005, +0.0007]
75         SP  large 0.05     0.0301  1.0241     0.0542  0.5011       0.4524  [+0.0001, +0.0014]
76         SP    LO small     0.0289  0.9848     0.0544  0.4348       0.4758  [-0.0010, +0.0000]
77         SP    LO large     0.0282  0.9604     0.0527  0.4877       0.4759  [-0.0018, -0.0005]
78         R4         own     0.0343  1.0000     0.0630  0.2510       0.6761                 NaN
79         R4       small     0.0348  1.0145     0.0673  0.0397       0.6853  [-0.0000, +0.0011]
80         R4       large     0.0313  0.9130     0.0571  0.3424       0.6873  [-0.0038, -0.0022]
81         R4  large 0.05     0.0311  0.9086     0.0570  0.3525       0.6872  [-0.0039, -0.0024]
82         R4    LO small     0.0334  0.9757     0.0634  0.0899       0.7017  [-0.0014, -0.0002]
83         R4    LO large     0.0308  0.8976     0.0566  0.3287       0.7022  [-0.0041, -0.0029]
84         R3         own     0.0348  1.0000     0.0627  0.1917       0.6960                 NaN
85         R3       small     0.0373  1.0715     0.0752 -0.1272       0.7315  [+0.0018, +0.0033]
86         R3       large     0.0322  0.9247     0.0583  0.3539       0.7325  [-0.0033, -0.0018]
87         R3  large 0.05     0.0317  0.9115     0.0579  0.3658       0.7324  [-0.0038, -0.0023]
88         R3    LO small     0.0360  1.0345     0.0705 -0.0698       0.7295  [+0.0005, +0.0020]
89         R3    LO large     0.0322  0.9252     0.0595  0.2871       0.7302  [-0.0032, -0.0020]
```

## Equations

small:

```
H_blue' = -0.1769·R·rate
H_red' = -0.1831·R·rate
p blue: 0.649 +0.000·focus
p red: 0.563 +0.201·focus
d' / v = -0.446·walk_r*wd +0.732·danger*wd -5.340·close*adv -2.798·close*hold*far -4.974·close*hold*contact
```

large:

```
H_blue' = -0.2907·R·1 +0.2659·R·ihp +0.3183·R·icd -0.2830·R·icd·ihp +0.1030·R·dmg -0.0723·R·dmg·ihp -0.1133·R·dmg·icd -0.1267·R·rate -0.0964·A·1 +0.0837·A·ihp +0.1183·A·icd -0.1057·A·icd·ihp +0.0444·A·dmg -0.0476·A·dmg·ihp -0.0703·A·dmg·icd +0.0309·A·rate +0.0332·A·rate·range -0.1912·Rfocus·1 +0.1852·Rfocus·ihp +0.2115·Rfocus·icd -0.2014·Rfocus·icd·ihp +0.0806·Rfocus·dmg -0.0842·Rfocus·dmg·ihp -0.1060·Rfocus·dmg·icd +0.0602·Rfocus·rate +0.0462·Rfocus·rate·range -0.1578·Rhold·1 +0.1608·Rhold·ihp +0.1766·Rhold·icd -0.1807·Rhold·icd·ihp +0.0636·Rhold·dmg -0.0615·Rhold·dmg·ihp -0.0800·Rhold·dmg·icd +0.0759·Rhold·rate -0.1510·R2·1 +0.0873·R2·ihp +0.3007·R2·icd -0.3202·R2·icd·ihp +0.1635·R2·dmg -0.0944·R2·dmg·ihp -0.3796·R2·dmg·icd +0.4079·R2·rate +0.0509·R2·rate·range
H_red' = -0.3440·R·1 +0.3243·R·ihp +0.3836·R·icd -0.3653·R·icd·ihp +0.1172·R·dmg -0.0777·R·dmg·ihp -0.1421·R·dmg·icd -0.1656·R·rate +0.0497·R·rate·range -0.1020·A·1 +0.0795·A·ihp +0.1336·A·icd -0.1097·A·icd·ihp +0.0551·A·dmg -0.0462·A·dmg·ihp -0.0869·A·dmg·icd +0.0628·A·rate -0.1567·Rfocus·1 +0.1365·Rfocus·ihp +0.1701·Rfocus·icd -0.1461·Rfocus·icd·ihp +0.0629·Rfocus·dmg -0.0506·Rfocus·dmg·ihp -0.0744·Rfocus·dmg·icd +0.1632·Rfocus·rate -0.0891·Rfocus·rate·range -0.1625·Rhold·1 +0.1632·Rhold·ihp +0.1812·Rhold·icd -0.1837·Rhold·icd·ihp +0.0690·Rhold·dmg -0.0752·Rhold·dmg·ihp -0.0880·Rhold·dmg·icd +0.0963·Rhold·rate -0.2055·R2·1 +0.3330·R2·icd -0.1913·R2·icd·ihp +0.1401·R2·dmg +0.0535·R2·dmg·ihp -0.3351·R2·dmg·icd +0.3320·R2·rate -0.0563·R2·rate·range
p blue: 0.633 +0.000·focus
p red: 0.522 +0.266·focus
d' / v = -0.446·walk_r*wd +0.732·danger*wd -5.340·close*adv -2.798·close*hold*far -4.974·close*hold*contact
```

large 0.05:

```
H_blue' = -0.3092·R·1 +0.2662·R·ihp +0.3375·R·icd -0.2859·R·icd·ihp +0.1095·R·dmg -0.0620·R·dmg·ihp -0.1225·R·dmg·icd -0.1075·R·rate -0.0265·R·rate·range -0.0889·A·1 +0.0769·A·ihp +0.1140·A·icd -0.1017·A·icd·ihp +0.0443·A·dmg -0.0479·A·dmg·ihp -0.0732·A·dmg·icd +0.0399·A·rate +0.0370·A·rate·range -0.0094·A·rate·speed -0.1935·Rfocus·1 +0.1891·Rfocus·ihp +0.2140·Rfocus·icd -0.2054·Rfocus·icd·ihp +0.0815·Rfocus·dmg -0.0865·Rfocus·dmg·ihp -0.1069·Rfocus·dmg·icd +0.0604·Rfocus·rate +0.0483·Rfocus·rate·range -0.1641·Rhold·1 +0.1670·Rhold·ihp +0.1834·Rhold·icd -0.1876·Rhold·icd·ihp +0.0661·Rhold·dmg -0.0639·Rhold·dmg·ihp -0.0831·Rhold·dmg·icd +0.0789·Rhold·rate -0.1366·R2·1 +0.0943·R2·ihp +0.2856·R2·icd -0.3241·R2·icd·ihp +0.1569·R2·dmg -0.1075·R2·dmg·ihp -0.3688·R2·dmg·icd +0.3887·R2·rate +0.0785·R2·rate·range
H_red' = -0.3240·R·1 +0.3238·R·ihp +0.3583·R·icd -0.3587·R·icd·ihp +0.1088·R·dmg -0.0879·R·dmg·ihp -0.1279·R·dmg·icd -0.2060·R·rate +0.0781·R·rate·range +0.0160·R·rate·speed -0.1194·A·1 +0.0827·A·ihp +0.1559·A·icd -0.1189·A·icd·ihp +0.0642·A·dmg -0.0421·A·dmg·ihp -0.1024·A·dmg·icd +0.0899·A·rate -0.0169·A·rate·range -0.0076·A·rate·speed -0.1485·Rfocus·1 +0.1270·Rfocus·ihp +0.1610·Rfocus·icd -0.1356·Rfocus·icd·ihp +0.0592·Rfocus·dmg -0.0454·Rfocus·dmg·ihp -0.0696·Rfocus·dmg·icd +0.1596·Rfocus·rate -0.0918·Rfocus·rate·range -0.1501·Rhold·1 +0.1509·Rhold·ihp +0.1678·Rhold·icd -0.1703·Rhold·icd·ihp +0.0640·Rhold·dmg -0.0701·Rhold·dmg·ihp -0.0818·Rhold·dmg·icd +0.0901·Rhold·rate -0.2257·R2·1 +0.0215·R2·ihp +0.3573·R2·icd -0.2157·R2·icd·ihp +0.1462·R2·dmg +0.0474·R2·dmg·ihp -0.3423·R2·dmg·icd +0.3736·R2·rate -0.0600·R2·rate·range -0.0302·R2·rate·speed
p blue: 0.633 +0.000·focus
p red: 0.522 +0.266·focus
d' / v = -0.446·walk_r*wd +0.732·danger*wd -5.340·close*adv -2.798·close*hold*far -4.974·close*hold*contact
```

LO small:

```
H_blue' = -0.1749·R·rate
H_red' = -0.1810·R·rate
p blue: 0.673 +0.000·focus
p red: 0.588 +0.226·focus
d' / v = -0.605·walk_r*wd +0.669·danger*wd -5.590·close*adv -2.941·close*hold*far -5.417·close*hold*contact
```

LO large:

```
H_blue' = +0.0574·R·1 -0.0538·R·icd·ihp -0.0766·R·dmg·ihp -0.0485·R·dmg·icd -0.1021·R·rate -0.0693·R2·icd +0.1501·R2·rate·range
H_red' = +0.0825·R·1 -0.0751·R·icd·ihp -0.0932·R·dmg·ihp -0.0660·R·dmg·icd -0.1395·R·rate +0.0462·R·rate·range +0.1123·Rfocus·rate -0.0958·Rfocus·rate·range -0.0713·R2·1 +0.0686·R2·dmg·ihp +0.1294·R2·rate -0.0475·R2·rate·range
p blue: 0.652 +0.000·focus
p red: 0.535 +0.313·focus
d' / v = -0.605·walk_r*wd +0.669·danger*wd -5.590·close*adv -2.941·close*hold*far -5.417·close*hold*contact
```
