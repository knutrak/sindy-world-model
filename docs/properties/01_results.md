# Unit properties, one at a time: results

Plan: `00_plan.md`. Script:
`experiments/properties/properties.py`. Run once.

## Rules

```
                                          result   met
P1 firepower                 15 of 16 (needs 14)  True
P2 no effect                    7 of 8 (needs 7)  True
P3 speed                      4 of 4 (needs all)  True
P4 rescaled beats unchanged     8 of 8 (needs 8)  True
P5 rescaled vs refit            8 of 8 (needs 7)  True
```

## Firepower (refits on 500 battles)

```
   variant losing side  k refit  ratio to V0   rule  ratio / rule other terms kept
0       V0        blue    0.179        1.000  1.000         1.000                -
1       V0         red    0.188        1.000  1.000         1.000                -
2       DM        blue    0.094        0.526  0.500         1.052                -
3       DM         red    0.100        0.533  0.500         1.066                -
4       DP        blue    0.259        1.452  2.000         0.726                1
5       DP         red    0.330        1.751  2.000         0.875                -
6       CM        blue    0.281        1.576  1.667         0.945                -
7       CM         red    0.302        1.605  1.667         0.963                -
8       CP        blue    0.095        0.530  0.500         1.061                -
9       CP         red    0.098        0.519  0.500         1.039                -
10      HM        blue    0.249        1.393  1.500         0.929                -
11      HM         red    0.266        1.411  1.500         0.940                -
12      HP        blue    0.094        0.526  0.500         1.052                -
13      HP         red    0.100        0.533  0.500         1.066                -
14      A1        blue    0.152        0.853  0.833         1.024                -
15      A1         red    0.161        0.853  0.833         1.024                -
16      A3        blue    0.094        0.526  0.500         1.052                -
17      A3         red    0.100        0.533  0.500         1.066                -
18      SM        blue    0.175        0.981  1.000         0.981                -
19      SM         red    0.190        1.008  1.000         1.008                -
20      SP        blue    0.178        0.997  1.000         0.997                -
21      SP         red    0.187        0.994  1.000         0.994                -
22      R4        blue    0.200        1.122  1.000         1.122                -
23      R4         red    0.175        0.932  1.000         0.932                -
24      R3        blue    0.205        1.148  1.000         1.148                -
25      R3         red    0.157        0.836  1.000         0.836                1
```

## Speed (refits)

```
  variant            term  refit     V0  ratio / rule
0      SM       close*adv -3.914 -6.020         0.975
1      SM  close*hold*far -2.131 -3.089         1.035
2      SP       close*adv -9.501 -6.020         1.052
3      SP  close*hold*far -4.754 -3.089         1.026
```

## Prediction (500 other battles; health in ten full units of the variant)

```
   rescaled - unchanged, health 2 s, paired 95% rescaled / refit, health 2 s
V0                           [-0.0000, +0.0000]                        1.002
DM                           [-0.0405, -0.0367]                      0.96453
DP                           [-0.0317, -0.0235]                     1.034455
CM                           [-0.0233, -0.0180]                       1.0532
CP                           [-0.0365, -0.0330]                      0.97733
HM                           [-0.0129, -0.0088]                     1.043883
HP                           [-0.0405, -0.0367]                      0.96453
A1                           [-0.0062, -0.0051]                     0.977424
A3                           [-0.0405, -0.0367]                      0.96453
SM                           [-0.0010, -0.0003]                     1.025874
SP                           [-0.0010, -0.0003]                     0.996289
R4                           [+0.0081, +0.0103]                     1.199063
R3                           [+0.0132, +0.0164]                     1.417516
```

```
   variant        model  health_0.5s  health_1s  health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_2s  distance_2s
0       V0  persistence       0.0296     0.0588     0.1163     0.2186       -0.0318       -0.0427    1.4683       0.6130
1       V0    unchanged       0.0111     0.0161     0.0276     0.0524        0.3360       -0.0161    0.5668       0.3855
2       V0     rescaled       0.0111     0.0161     0.0276     0.0524        0.3360       -0.0161    0.5668       0.3855
3       V0        refit       0.0111     0.0160     0.0276     0.0520        0.3614        0.0403    0.5631       0.3845
4       DM  persistence       0.0165     0.0330     0.0657     0.1289       -0.0978       -0.1266    0.8122       0.4939
5       DM    unchanged       0.0147     0.0280     0.0525     0.0859       -0.7322       -1.1469    0.8076       0.3659
6       DM     rescaled       0.0056     0.0082     0.0140     0.0281        0.5645        0.3633    0.4453       0.3653
7       DM        refit       0.0055     0.0082     0.0145     0.0296        0.5297        0.3203    0.4405       0.3372
8       DP  persistence       0.0470     0.0937     0.1780     0.2709       -0.0024       -0.0003    2.2689       0.8350
9       DP    unchanged       0.0271     0.0459     0.0781     0.1096        0.3504        0.0179    1.1591       0.4495
10      DP     rescaled       0.0205     0.0297     0.0508     0.0884       -0.0043       -0.5864    0.7958       0.4502
11      DP        refit       0.0212     0.0297     0.0491     0.0895        0.0476       -0.6098    0.7942       0.4505
12      CM  persistence       0.0429     0.0860     0.1669     0.2739       -0.0219       -0.0148    2.1155       0.7480
13      CM    unchanged       0.0189     0.0339     0.0602     0.0921        0.4555        0.1068    0.9471       0.3950
14      CM     rescaled       0.0131     0.0206     0.0397     0.0733        0.1922       -0.3299    0.6843       0.3953
15      CM        refit       0.0129     0.0202     0.0377     0.0689        0.2542       -0.2083    0.6657       0.3986
16      CP  persistence       0.0166     0.0331     0.0658     0.1287       -0.0489       -0.0744    0.8222       0.4842
17      CP    unchanged       0.0177     0.0295     0.0538     0.0863       -0.7844       -1.2221    0.8014       0.3650
18      CP     rescaled       0.0113     0.0125     0.0191     0.0322        0.4050        0.2939    0.4597       0.3644
19      CP        refit       0.0113     0.0125     0.0196     0.0336        0.3734        0.2492    0.4588       0.3390
20      HM  persistence       0.0390     0.0778     0.1509     0.2564       -0.0056       -0.0047    1.9309       0.7226
21      HM    unchanged       0.0188     0.0303     0.0510     0.0802        0.3660        0.0433    0.8111       0.4093
22      HM     rescaled       0.0161     0.0234     0.0402     0.0720        0.1137       -0.3542    0.6761       0.4095
23      HM        refit       0.0160     0.0230     0.0385     0.0686        0.2018       -0.2207    0.6557       0.4171
24      HP  persistence       0.0165     0.0330     0.0657     0.1289       -0.0978       -0.1266    0.8122       0.4939
25      HP    unchanged       0.0147     0.0280     0.0525     0.0859       -0.7322       -1.1469    0.8076       0.3659
26      HP     rescaled       0.0056     0.0082     0.0140     0.0281        0.5645        0.3633    0.4453       0.3653
27      HP        refit       0.0055     0.0082     0.0145     0.0296        0.5297        0.3203    0.4405       0.3372
28      A1  persistence       0.0253     0.0504     0.0999     0.1907       -0.0450       -0.0597    1.2521       0.5666
29      A1    unchanged       0.0096     0.0153     0.0282     0.0526        0.3201       -0.0219    0.5433       0.3707
30      A1     rescaled       0.0091     0.0133     0.0226     0.0436        0.4769        0.1899    0.5054       0.3704
31      A1        refit       0.0091     0.0133     0.0231     0.0448        0.4261        0.1012    0.5095       0.3653
32      A3  persistence       0.0165     0.0330     0.0657     0.1289       -0.0978       -0.1266    0.8122       0.4939
33      A3    unchanged       0.0147     0.0280     0.0525     0.0859       -0.7322       -1.1469    0.8076       0.3659
34      A3     rescaled       0.0056     0.0082     0.0140     0.0281        0.5645        0.3633    0.4453       0.3653
35      A3        refit       0.0055     0.0082     0.0145     0.0296        0.5297        0.3203    0.4405       0.3372
36      SM  persistence       0.0277     0.0557     0.1118     0.2190       -0.0419       -0.0549    1.3856       0.5885
37      SM    unchanged       0.0106     0.0150     0.0269     0.0521        0.2997       -0.1185    0.5643       0.3247
38      SM     rescaled       0.0107     0.0153     0.0262     0.0503        0.2885       -0.1179    0.5484       0.3034
39      SM        refit       0.0107     0.0151     0.0256     0.0483        0.2978       -0.0855    0.5471       0.2958
40      SP  persistence       0.0314     0.0616     0.1193     0.2146       -0.0230       -0.0300    1.5280       0.6379
41      SP    unchanged       0.0119     0.0174     0.0300     0.0571        0.4087        0.0124    0.5662       0.4575
42      SP     rescaled       0.0118     0.0170     0.0293     0.0569        0.4162        0.0133    0.5647       0.4763
43      SP        refit       0.0118     0.0170     0.0294     0.0558        0.4276        0.0507    0.5581       0.4803
44      R4  persistence       0.0311     0.0623     0.1241     0.2329       -0.2315       -0.2275    1.5897       0.9436
45      R4    unchanged       0.0116     0.0177     0.0319     0.0643        0.0895       -0.4440    0.5966       0.7840
46      R4     rescaled       0.0123     0.0208     0.0411     0.0820        0.0931       -0.2751    0.6692       0.7706
47      R4        refit       0.0118     0.0189     0.0343     0.0630        0.2510        0.0206    0.6042       0.6761
48      R3  persistence       0.0330     0.0661     0.1329     0.2527       -0.3817       -0.4150    1.6859       1.0485
49      R3    unchanged       0.0117     0.0181     0.0345     0.0728       -0.1041       -0.7779    0.6502       0.9135
50      R3     rescaled       0.0126     0.0225     0.0493     0.1114       -0.0242       -0.3608    0.7440       0.8925
51      R3        refit       0.0122     0.0193     0.0348     0.0627        0.1917       -0.0284    0.6328       0.6960
```

## Refitted equations

V0:

```
H_B^marine_v0' = -0.1785·R[marine_v0]*s[marine_v0]
H_R^marine_v0' = -0.1883·R[marine_v0]*s[marine_v0]
(ln A_B^marine_v0)' = (0.659) · (ln H_B^marine_v0)'
(ln A_R^marine_v0)' = (0.550 +0.284·focus) · (ln H_R^marine_v0)'
d' = -0.577·walk_r*wd +0.656·danger*wd -6.020·close*adv -3.089·close*hold*far -6.350·close*hold*contact
```

DM:

```
H_B^marine_dm' = -0.0939·R[marine_dm]*s[marine_dm]
H_R^marine_dm' = -0.1004·R[marine_dm]*s[marine_dm]
(ln A_B^marine_dm)' = (0.622) · (ln H_B^marine_dm)'
(ln A_R^marine_dm)' = (0.493 +0.324·focus) · (ln H_R^marine_dm)'
d' = -0.310·walk_r*wd +0.514·danger*wd -5.955·close*adv -3.070·close*hold*far -6.232·close*hold*contact
```

DP:

```
H_B^marine_dp' = -0.2593·R[marine_dp]*s[marine_dp] -0.0232·1
H_R^marine_dp' = -0.3297·R[marine_dp]*s[marine_dp]
(ln A_B^marine_dp)' = (0.692) · (ln H_B^marine_dp)'
(ln A_R^marine_dp)' = (0.645 +0.168·focus) · (ln H_R^marine_dp)'
d' = -0.813·walk_r*wd +0.745·danger*wd -6.073·close*adv -3.133·close*hold*far -6.146·close*hold*contact
```

CM:

```
H_B^marine_cm' = -0.2813·R[marine_cm]*s[marine_cm]
H_R^marine_cm' = -0.3023·R[marine_cm]*s[marine_cm]
(ln A_B^marine_cm)' = (0.681) · (ln H_B^marine_cm)'
(ln A_R^marine_cm)' = (0.588 +0.212·focus) · (ln H_R^marine_cm)'
d' = -0.772·walk_r*wd +0.730·danger*wd -6.059·close*adv -3.196·close*hold*far -6.166·close*hold*contact
```

CP:

```
H_B^marine_cp' = -0.0947·R[marine_cp]*s[marine_cp]
H_R^marine_cp' = -0.0978·R[marine_cp]*s[marine_cp]
(ln A_B^marine_cp)' = (0.633) · (ln H_B^marine_cp)'
(ln A_R^marine_cp)' = (0.524 +0.306·focus) · (ln H_R^marine_cp)'
d' = -0.287·walk_r*wd +0.502·danger*wd -5.938·close*adv -3.197·close*hold*far -6.234·close*hold*contact
```

HM:

```
H_B^marine_hm' = -0.2487·R[marine_hm]*s[marine_hm]
H_R^marine_hm' = -0.2657·R[marine_hm]*s[marine_hm]
(ln A_B^marine_hm)' = (0.680) · (ln H_B^marine_hm)'
(ln A_R^marine_hm)' = (0.595 +0.208·focus) · (ln H_R^marine_hm)'
d' = -0.745·walk_r*wd +0.716·danger*wd -6.025·close*adv -3.154·close*hold*far -6.120·close*hold*contact
```

HP:

```
H_B^marine_hp' = -0.0939·R[marine_hp]*s[marine_hp]
H_R^marine_hp' = -0.1004·R[marine_hp]*s[marine_hp]
(ln A_B^marine_hp)' = (0.622) · (ln H_B^marine_hp)'
(ln A_R^marine_hp)' = (0.493 +0.324·focus) · (ln H_R^marine_hp)'
d' = -0.310·walk_r*wd +0.514·danger*wd -5.955·close*adv -3.070·close*hold*far -6.232·close*hold*contact
```

A1:

```
H_B^marine_a1' = -0.1523·R[marine_a1]*s[marine_a1]
H_R^marine_a1' = -0.1606·R[marine_a1]*s[marine_a1]
(ln A_B^marine_a1)' = (0.649) · (ln H_B^marine_a1)'
(ln A_R^marine_a1)' = (0.544 +0.286·focus) · (ln H_R^marine_a1)'
d' = -0.533·walk_r*wd +0.631·danger*wd -5.983·close*adv -3.090·close*hold*far -6.443·close*hold*contact
```

A3:

```
H_B^marine_a3' = -0.0939·R[marine_a3]*s[marine_a3]
H_R^marine_a3' = -0.1004·R[marine_a3]*s[marine_a3]
(ln A_B^marine_a3)' = (0.622) · (ln H_B^marine_a3)'
(ln A_R^marine_a3)' = (0.493 +0.324·focus) · (ln H_R^marine_a3)'
d' = -0.310·walk_r*wd +0.514·danger*wd -5.955·close*adv -3.070·close*hold*far -6.232·close*hold*contact
```

SM:

```
H_B^marine_sm' = -0.1752·R[marine_sm]*s[marine_sm]
H_R^marine_sm' = -0.1899·R[marine_sm]*s[marine_sm]
(ln A_B^marine_sm)' = (0.661) · (ln H_B^marine_sm)'
(ln A_R^marine_sm)' = (0.562 +0.253·focus) · (ln H_R^marine_sm)'
d' = -0.689·walk_r*wd +0.624·danger*wd -3.914·close*adv -2.131·close*hold*far -3.814·close*hold*contact
```

SP:

```
H_B^marine_sp' = -0.1780·R[marine_sp]*s[marine_sp]
H_R^marine_sp' = -0.1872·R[marine_sp]*s[marine_sp]
(ln A_B^marine_sp)' = (0.660) · (ln H_B^marine_sp)'
(ln A_R^marine_sp)' = (0.566 +0.256·focus) · (ln H_R^marine_sp)'
d' = -0.458·walk_r*wd +0.610·danger*wd -9.501·close*adv -4.754·close*hold*far -10.382·close*hold*contact
```

R4:

```
H_B^marine_r4' = -0.2003·R[marine_r4]*s[marine_r4]
H_R^marine_r4' = -0.1755·R[marine_r4]*s[marine_r4]
(ln A_B^marine_r4)' = (0.683) · (ln H_B^marine_r4)'
(ln A_R^marine_r4)' = (0.658 +0.139·focus) · (ln H_R^marine_r4)'
d' = -1.139·walk_r*wd +1.291·danger*wd -4.522·close*adv -2.288·close*hold*far -4.010·close*hold*contact
```

R3:

```
H_B^marine_r3' = -0.2049·R[marine_r3]*s[marine_r3]
H_R^marine_r3' = -0.1574·R[marine_r3]*s[marine_r3] -0.0102·1
(ln A_B^marine_r3)' = (0.661) · (ln H_B^marine_r3)'
(ln A_R^marine_r3)' = (0.615 +0.196·focus) · (ln H_R^marine_r3)'
d' = -1.657·walk_r*wd +1.596·danger*wd -5.035·close*adv -2.971·close*hold*far -5.233·close*hold*contact
```

## Notes added after the run (25 Nov)

- **Duplicates, a flaw in the plan.** DM, A3 and HP are the same battles (checked on 50 battles: positions and hp fractions
  identical), since the game only sees net damage per hp. The plan called them separate routes to the ratio 0.50; they
  are one. The firepower variants are therefore 6, not 8: P1 11 of 12, P4 6 of 6, P5 6 of 6. No rule was changed.
- **Misses.** P1: DP, Blue's losses, 0.73 × the rule (the refit also keeps a constant). P2: R3, Red's losses, 0.84.
- **Pattern in P1.** Ratio / rule is 1.02-1.07 in the low-firepower cases (DM, CP, A1) and 0.73-0.96 in the
  high-firepower ones (CM, HM, DP): firepower rises less than the game rate.
- **Range, reported and not judged.** Moving only ρ made predictions worse than leaving the model unchanged (2 s: R4 0.041
  against 0.032, R3 0.049 against 0.034; 1.20 and 1.42 × the refit). The gates were left unchanged, as the plan said.
- **Margin R², reported and not judged.** In DP, CM and HM the rescaled and refitted models have lower margin R² at 2 s than
  the unchanged model (e.g. CM 0.19 and 0.25 against 0.46), although their health errors are lower.
- Knut did not write an expectation; the rules are Claude's proposals, approved by Knut before the run.
