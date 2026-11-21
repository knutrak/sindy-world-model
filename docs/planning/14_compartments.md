# Health compartments in the world model: results

Plan: `13_compartment_plan.md` (pre-registered, approved 21 Nov). Script: `experiments/planning/compartment_model.py`.
States: `scripts/build_compartment_states.py`. Fitted on train only; tested on val, the counterfactual
sets and 400 fresh seeds per scenario (16 x00 000 +).

## The fitted laws (train)

```
S1 (marines model)
I_B^marine = d(1-w)·(-4.0287 +0.4227·focus +0.0285·hold +2.3607·wd) + d(1-w)/mF·(+11.1714 -1.0226·focus -0.0100·hold -6.0206·wd) + d(1-w)rho·(-1.1418 +0.0868·focus -0.0315·hold +0.5836·wd)
KW_B^marine = dw·(+1.7810 +0.1074·focus +0.9699·hold +0.1500·wd) + dw/mW·(+0.2049 -0.0284·focus -0.0072·hold -0.0459·wd) + W·(+0.3178 -0.0205·focus -0.1407·hold -0.2430·wd)
KF_B^marine = d(1-w)·(+0.8765 +0.0565·focus +0.0327·hold -0.2727·wd) + d(1-w)/mF·(-0.6133 -0.0147·focus -0.0410·hold +0.2949·wd)
sigma_B^marine = w·(+1.9050 +0.0564·focus +0.0636·hold +0.3893·wd) + w2·(-0.5862 -0.0396·focus -0.0640·hold -0.5746·wd) + w(1-mW)·(-0.9309 -0.0296·focus -0.0475·hold +0.2762·wd)
I_R^marine = d(1-w)·(-2.0438 +1.1476·focus -0.0084·hold) + d(1-w)/mF·(+7.4889 -4.1471·focus +0.1205·hold) + d(1-w)rho·(-1.2887 +0.8691·focus -0.0694·hold)
KW_R^marine = dw·(+1.0334 +1.0196·focus +1.0146·hold) + dw/mW·(+0.1350 -0.0783·focus -0.0112·hold) + W·(+0.3326 +0.0868·focus -0.1696·hold -0.3759·wd)
KF_R^marine = d(1-w)·(+0.6347 -0.8985·focus -0.1236·hold) + d(1-w)/mF·(-0.0776 +2.2337·focus +0.2370·hold)
sigma_R^marine = w·(+1.3515 +0.9387·focus +0.0895·hold) + w2·(-0.0742 -0.7799·focus -0.1057·hold) + w(1-mW)·(-0.8142 -0.5396·focus -0.0770·hold)

S2 (mixed model)
I_B^marauder = d(1-w)·(-5.1478 +0.2507·focus +0.0582·hold +0.6211·wd) + d(1-w)/mF·(+13.8649 -0.4236·focus +0.2792·hold -2.1394·wd) + d(1-w)rho·(-1.1989 +0.0369·focus -0.2881·hold +0.5581·wd) + dw·(-0.4507 -0.0702·focus -0.0162·hold -0.2193·wd)
KW_B^marauder = dw·(+1.4826 -0.0378·focus +0.3985·hold +0.6594·wd) + dw/mW·(+0.3569 +0.0119·focus -0.0036·hold -0.0616·wd) + W·(+0.1231 +0.0086·focus -0.0413·hold -0.0762·wd)
KF_B^marauder = d(1-w)·(+0.4179 +0.0062·focus +0.0762·hold -0.1062·wd) + d(1-w)/mF·(-0.3725 +0.0211·focus -0.1314·hold +0.1735·wd)
sigma_B^marauder = w·(+2.1337 +0.0821·focus +0.1374·hold +0.2210·wd) + w2·(-1.0645 -0.0681·focus -0.1207·hold -0.2861·wd) + w(1-mW)·(-0.2181 -0.0203·focus -0.0471·hold +0.1216·wd)
I_B^marine = d(1-w)·(-3.9342 -0.0580·focus +0.4709·hold +0.4925·wd) + d(1-w)/mF·(+10.6345 -0.1140·focus -0.8377·hold -2.6591·wd) + d(1-w)rho·(-0.8594 +0.1142·focus -0.0450·hold +0.7078·wd)
KW_B^marine = dw·(+1.7980 -0.0130·focus +0.6267·hold +0.3719·wd) + dw/mW·(+0.1188 -0.0136·focus +0.0075·hold +0.0230·wd) + W·(+0.2514 +0.0107·focus -0.0934·hold -0.2202·wd)
KF_B^marine = d(1-w)·(+0.3000 +0.1465·focus -0.1366·hold +0.0798·wd) + d(1-w)/mF·(+0.3204 -0.1932·focus +0.2586·hold -0.3844·wd)
sigma_B^marine = w·(+2.0218 +0.0234·focus +0.0709·hold +0.2740·wd) + w2·(-0.8713 -0.0131·focus -0.0546·hold -0.3949·wd) + w(1-mW)·(-0.3669 -0.0230·focus -0.0436·hold +0.1452·wd)
I_R^marauder = d(1-w)·(-4.1790 +0.7561·focus -0.0562·hold) + d(1-w)/mF·(+11.4782 -2.2454·focus +0.2002·hold) + d(1-w)rho·(-0.8861 +0.6277·focus -0.0957·hold) + dw·(-0.3023 -0.1699·focus +0.0314·hold)
KW_R^marauder = dw·(+1.0308 +0.5746·focus +0.7124·hold) + dw/mW·(+0.3341 -0.0391·focus -0.0056·hold) + W·(+0.1505 +0.0370·focus -0.0894·hold -0.1698·wd)
KF_R^marauder = d(1-w)·(+0.3751 -0.7441·focus -0.0691·hold) + d(1-w)/mF·(-0.2494 +1.2326·focus +0.1024·hold)
sigma_R^marauder = w·(+1.7108 +0.6477·focus +0.0415·hold) + w2·(-0.5935 -0.6557·focus -0.0574·hold) + w(1-mW)·(-0.3281 -0.0442·focus +0.0168·hold)
I_R^marine = d(1-w)·(-0.8143 +1.1938·focus +0.1153·hold) + d(1-w)/mF·(+4.7127 -3.4225·focus -0.1154·hold) + d(1-w)rho·(-1.0378 +0.5486·focus -0.0312·hold)
KW_R^marine = dw·(+1.2131 -0.0268·focus +0.4588·hold) + dw/mW·(+0.0567 -0.0314·focus -0.0075·hold) + W·(+0.3055 +0.1525·focus -0.0983·hold -0.3761·wd)
KF_R^marine = d(1-w)·(+0.0183 -0.0283·focus -0.1155·hold) + d(1-w)/mF·(+1.0342 +0.5077·focus +0.1414·hold)
sigma_R^marine = w·(+1.4337 +0.3063·focus +0.1036·hold) + w2·(-0.2704 -0.2714·focus -0.0779·hold) + w(1-mW)·(-0.3674 -0.0736·focus -0.0617·hold)

```

## H1: CV R² per law (train, 5 folds by battle)

```
variant                       compartments  without H_W
scenario side type     law                             
S1       blue marine   I            0.2485       0.1050
                       KF           0.3227       0.3217
                       KW           0.5751       0.5619
                       sigma        0.6113       0.5770
         red  marine   I            0.2635       0.1986
                       KF           0.4333       0.4277
                       KW           0.6990       0.6929
                       sigma        0.6293       0.5744
S2       blue marauder I            0.3808       0.2147
                       KF           0.0822       0.0805
                       KW           0.4126       0.3543
                       sigma        0.7333       0.7308
              marine   I            0.2478       0.1209
                       KF           0.2411       0.2411
                       KW           0.5254       0.5165
                       sigma        0.6286       0.6219
         red  marauder I            0.3728       0.2460
                       KF           0.0782       0.0719
                       KW           0.4858       0.4392
                       sigma        0.7135       0.7082
              marine   I            0.1576       0.1308
                       KF           0.3672       0.3577
                       KW           0.6391       0.6371
                       sigma        0.6734       0.6619
```

## H2: kills over 1 s windows (CV R², d-hat for both)

```
                        kills CV R², compartments  frozen law (refit)
scenario side type                                                   
S1       blue marine                       0.4910              0.4806
         red  marine                       0.6082              0.6024
S2       blue marauder                     0.3931              0.2978
              marine                       0.4739              0.4365
         red  marauder                     0.4531              0.3613
              marine                       0.5699              0.5554
```

## H3: alive error on val (team, units; paired, bootstrap by battle)

```
                                                                  alive error, frozen  alive error  difference                 95%       read-out  health error difference          health 95%
scenario subset                    h   model                                                                                                                                                  
S1       both sides started >= 80% 1.0 compartments                            0.7660       0.7804      0.0144  [-0.0079, +0.0367]  no difference                   0.0000  [-0.0001, +0.0002]
                                       compartments + focus cost               0.7660       0.7827      0.0167  [-0.0056, +0.0395]  no difference                   0.0000  [-0.0002, +0.0002]
                                       without H_W                             0.7660       0.7909      0.0249  [+0.0021, +0.0487]          worse                   0.0000  [-0.0001, +0.0002]
                                   2.0 compartments                            0.9859       1.0609      0.0749  [+0.0368, +0.1169]          worse                   0.0001  [-0.0005, +0.0007]
                                       compartments + focus cost               0.9859       1.0670      0.0811  [+0.0421, +0.1235]          worse                   0.0000  [-0.0007, +0.0007]
                                       without H_W                             0.9859       1.1091      0.1232  [+0.0823, +0.1646]          worse                   0.0000  [-0.0006, +0.0007]
                                   4.0 compartments                            1.5662       1.6589      0.0928  [+0.0169, +0.1699]          worse                   0.0002  [-0.0019, +0.0025]
                                       compartments + focus cost               1.5662       1.6676      0.1015  [+0.0231, +0.1801]          worse                  -0.0001  [-0.0025, +0.0023]
                                       without H_W                             1.5662       1.8049      0.2387  [+0.1609, +0.3301]          worse                   0.0015  [-0.0012, +0.0041]
         all starts                1.0 compartments                            0.8392       0.8308     -0.0085  [-0.0182, +0.0012]  no difference                   0.0000  [-0.0001, +0.0001]
                                       compartments + focus cost               0.8392       0.8346     -0.0047  [-0.0147, +0.0048]  no difference                  -0.0001  [-0.0001, +0.0000]
                                       without H_W                             0.8392       0.8417      0.0025  [-0.0078, +0.0129]  no difference                   0.0000  [-0.0000, +0.0001]
                                   2.0 compartments                            1.1010       1.1394      0.0384  [+0.0203, +0.0559]          worse                   0.0005  [+0.0002, +0.0007]
                                       compartments + focus cost               1.1010       1.1467      0.0457  [+0.0286, +0.0641]          worse                   0.0005  [+0.0002, +0.0008]
                                       without H_W                             1.1010       1.1886      0.0876  [+0.0693, +0.1061]          worse                   0.0005  [+0.0002, +0.0008]
                                   4.0 compartments                            1.8261       1.8396      0.0135  [-0.0231, +0.0497]  no difference                   0.0007  [-0.0002, +0.0016]
                                       compartments + focus cost               1.8261       1.8513      0.0253  [-0.0104, +0.0630]  no difference                   0.0005  [-0.0006, +0.0015]
                                       without H_W                             1.8261       1.9632      0.1371  [+0.0960, +0.1786]          worse                   0.0014  [+0.0004, +0.0025]
S2       both sides started >= 80% 1.0 compartments                            0.7035       0.7060      0.0024  [-0.0135, +0.0190]  no difference                   0.0002  [+0.0001, +0.0003]
                                       compartments + focus cost               0.7035       0.7062      0.0027  [-0.0133, +0.0177]  no difference                   0.0001  [+0.0000, +0.0002]
                                       without H_W                             0.7035       0.7182      0.0146  [-0.0010, +0.0291]  no difference                   0.0003  [+0.0002, +0.0003]
                                   2.0 compartments                            0.9351       0.9616      0.0265  [-0.0045, +0.0588]  no difference                   0.0004  [-0.0001, +0.0008]
                                       compartments + focus cost               0.9351       0.9637      0.0285  [-0.0021, +0.0595]  no difference                   0.0002  [-0.0002, +0.0007]
                                       without H_W                             0.9351       0.9790      0.0439  [+0.0155, +0.0729]          worse                   0.0004  [-0.0001, +0.0008]
                                   4.0 compartments                            1.3641       1.3885      0.0244  [-0.0344, +0.0826]  no difference                  -0.0001  [-0.0019, +0.0018]
                                       compartments + focus cost               1.3641       1.3921      0.0280  [-0.0280, +0.0817]  no difference                   0.0001  [-0.0017, +0.0019]
                                       without H_W                             1.3641       1.4716      0.1075  [+0.0511, +0.1655]          worse                  -0.0002  [-0.0021, +0.0018]
         all starts                1.0 compartments                            0.7574       0.7461     -0.0113  [-0.0195, -0.0031]         better                   0.0000  [-0.0000, +0.0001]
                                       compartments + focus cost               0.7574       0.7473     -0.0101  [-0.0184, -0.0021]         better                  -0.0001  [-0.0001, -0.0000]
                                       without H_W                             0.7574       0.7612      0.0038  [-0.0044, +0.0111]  no difference                   0.0001  [+0.0000, +0.0001]
                                   2.0 compartments                            1.0028       1.0041      0.0013  [-0.0136, +0.0165]  no difference                  -0.0001  [-0.0003, +0.0001]
                                       compartments + focus cost               1.0028       1.0065      0.0038  [-0.0106, +0.0187]  no difference                  -0.0002  [-0.0004, -0.0000]
                                       without H_W                             1.0028       1.0296      0.0269  [+0.0134, +0.0416]          worse                  -0.0000  [-0.0002, +0.0001]
                                   4.0 compartments                            1.4777       1.4663     -0.0113  [-0.0384, +0.0144]  no difference                  -0.0009  [-0.0016, -0.0002]
                                       compartments + focus cost               1.4777       1.4701     -0.0076  [-0.0335, +0.0201]  no difference                  -0.0008  [-0.0015, -0.0002]
                                       without H_W                             1.4777       1.5355      0.0578  [+0.0324, +0.0850]          worse                  -0.0010  [-0.0017, -0.0003]
```

## H4: ranking on the counterfactual sets

```
                                        regret  rank corr  always adv/nearest  picks focus  eventual won  eventual regret
scenario h   model                                                                                                       
S1       1.0 frozen                     0.0170     0.2386              0.0052       0.9847        0.5805           0.0839
             compartments               0.0169     0.2459              0.0052       0.9828        0.5805           0.0843
             compartments + focus cost  0.0055     0.5522              0.0052       0.0038        0.5747           0.0876
             without H_W                0.0168     0.2454              0.0052       0.9828        0.5805           0.0843
         2.0 frozen                     0.0225     0.3477              0.0109       0.9636        0.5824           0.0832
             compartments               0.0224     0.3564              0.0109       0.9598        0.5843           0.0826
             compartments + focus cost  0.0108     0.5315              0.0109       0.0556        0.5747           0.0865
             without H_W                0.0225     0.3552              0.0109       0.9540        0.5843           0.0826
         4.0 frozen                     0.0308     0.3873              0.0267       0.9195        0.5881           0.0828
             compartments               0.0309     0.3891              0.0267       0.9176        0.5900           0.0817
             compartments + focus cost  0.0253     0.4244              0.0267       0.1877        0.5862           0.0839
             without H_W                0.0309     0.3830              0.0267       0.9061        0.5900           0.0815
         8.0 frozen                     0.0580     0.1800              0.0608       0.9272        0.5785           0.0855
             compartments               0.0602     0.1645              0.0608       0.9042        0.5785           0.0872
             compartments + focus cost  0.0573     0.1504              0.0608       0.3142        0.5900           0.0839
             without H_W                0.0592     0.1772              0.0608       0.8927        0.5824           0.0865
S2       1.0 frozen                     0.0188     0.2971              0.0063       0.9716        0.6087           0.1581
             compartments               0.0184     0.3110              0.0063       0.9281        0.6104           0.1596
             compartments + focus cost  0.0062     0.5740              0.0063       0.0100        0.6020           0.1671
             without H_W                0.0185     0.3073              0.0063       0.9197        0.6054           0.1604
         2.0 frozen                     0.0297     0.3240              0.0121       0.9599        0.6120           0.1543
             compartments               0.0299     0.3248              0.0121       0.9498        0.6104           0.1556
             compartments + focus cost  0.0122     0.5779              0.0121       0.0368        0.5987           0.1656
             without H_W                0.0299     0.3255              0.0121       0.9398        0.6070           0.1570
         4.0 frozen                     0.0444     0.3477              0.0355       0.8930        0.6154           0.1525
             compartments               0.0445     0.3464              0.0355       0.8779        0.6154           0.1506
             compartments + focus cost  0.0372     0.4469              0.0355       0.1037        0.6104           0.1588
             without H_W                0.0438     0.3433              0.0355       0.8696        0.6120           0.1517
         8.0 frozen                     0.0747     0.2151              0.0884       0.8261        0.6288           0.1511
             compartments               0.0782     0.1798              0.0884       0.7977        0.6221           0.1555
             compartments + focus cost  0.0848     0.1863              0.0884       0.2759        0.6187           0.1554
             without H_W                0.0800     0.1905              0.0884       0.7860        0.6254           0.1570
```

```
                                        regret − frozen                 95%
scenario h   model                                                         
S1       1.0 compartments                       -0.0001  [-0.0002, -0.0000]
             compartments + focus cost          -0.0115  [-0.0135, -0.0094]
             without H_W                        -0.0001  [-0.0003, -0.0000]
         2.0 compartments                       -0.0001  [-0.0002, +0.0000]
             compartments + focus cost          -0.0117  [-0.0148, -0.0085]
             without H_W                        -0.0000  [-0.0002, +0.0002]
         4.0 compartments                        0.0002  [-0.0007, +0.0011]
             compartments + focus cost          -0.0055  [-0.0108, -0.0002]
             without H_W                         0.0002  [-0.0010, +0.0016]
         8.0 compartments                        0.0022  [+0.0007, +0.0040]
             compartments + focus cost          -0.0007  [-0.0075, +0.0057]
             without H_W                         0.0013  [-0.0005, +0.0033]
S2       1.0 compartments                       -0.0005  [-0.0009, -0.0001]
             compartments + focus cost          -0.0126  [-0.0148, -0.0104]
             without H_W                        -0.0003  [-0.0006, -0.0000]
         2.0 compartments                        0.0002  [-0.0002, +0.0008]
             compartments + focus cost          -0.0175  [-0.0213, -0.0137]
             without H_W                         0.0002  [-0.0003, +0.0008]
         4.0 compartments                        0.0001  [-0.0017, +0.0022]
             compartments + focus cost          -0.0072  [-0.0136, -0.0008]
             without H_W                        -0.0005  [-0.0022, +0.0012]
         8.0 compartments                        0.0035  [+0.0003, +0.0071]
             compartments + focus cost           0.0100  [+0.0009, +0.0199]
             without H_W                         0.0053  [+0.0013, +0.0092]
```

## H5: closed loop on fresh seeds (wins, paired)

```
                                                                       arm won  against won  difference               95% read-out    n  flagged
scenario subset                    arm                  against                                                                                 
S1       all starts                compart:h3           model:h3         0.678        0.675       0.002  [-0.010, +0.015]     ties  400    False
                                   compart-focuscost:h3 model:h3         0.680        0.675       0.005  [-0.037, +0.048]     ties  400    False
                                   model:h3             hold/focus       0.675        0.715      -0.040  [-0.072, -0.007]    loses  400    False
                                   compart:h3           hold/focus       0.678        0.715      -0.038  [-0.070, -0.007]    loses  400    False
                                   compart-focuscost:h3 hold/focus       0.680        0.715      -0.035  [-0.080, +0.010]     ties  400    False
         both sides started >= 80% compart:h3           model:h3         0.690        0.690       0.000  [+0.000, +0.000]     ties   71    False
                                   compart-focuscost:h3 model:h3         0.676        0.690      -0.014  [-0.099, +0.071]     ties   71    False
                                   model:h3             hold/focus       0.690        0.718      -0.028  [-0.099, +0.042]     ties   71    False
                                   compart:h3           hold/focus       0.690        0.718      -0.028  [-0.099, +0.042]     ties   71    False
                                   compart-focuscost:h3 hold/focus       0.676        0.718      -0.042  [-0.155, +0.056]     ties   71    False
S2       all starts                compart:h3           model:h3         0.650        0.658      -0.008  [-0.022, +0.007]     ties  400    False
                                   compart-focuscost:h3 model:h3         0.662        0.658       0.005  [-0.043, +0.052]     ties  400    False
                                   model:h3             advance/focus    0.658        0.655       0.002  [-0.018, +0.022]     ties  400    False
                                   compart:h3           advance/focus    0.650        0.655      -0.005  [-0.025, +0.015]     ties  400    False
                                   compart-focuscost:h3 advance/focus    0.662        0.655       0.008  [-0.040, +0.055]     ties  400    False
         both sides started >= 80% compart:h3           model:h3         0.711        0.711       0.000  [-0.039, +0.039]     ties   76    False
                                   compart-focuscost:h3 model:h3         0.697        0.711      -0.013  [-0.132, +0.105]     ties   76    False
                                   model:h3             advance/focus    0.711        0.711       0.000  [-0.039, +0.039]     ties   76    False
                                   compart:h3           advance/focus    0.711        0.711       0.000  [-0.039, +0.039]     ties   76    False
                                   compart-focuscost:h3 advance/focus    0.697        0.711      -0.013  [-0.132, +0.105]     ties   76    False
S3       all starts                compart:h3           model:h3         0.728        0.735      -0.008  [-0.037, +0.022]     ties  400     True
                                   compart-focuscost:h3 model:h3         0.745        0.735       0.010  [-0.022, +0.045]     ties  400     True
                                   model:h3             hold/focus       0.735        0.585       0.150  [+0.100, +0.203]    beats  400     True
                                   compart:h3           hold/focus       0.728        0.585       0.142  [+0.092, +0.193]    beats  400     True
                                   compart-focuscost:h3 hold/focus       0.745        0.585       0.160  [+0.107, +0.210]    beats  400     True
         both sides started >= 80% compart:h3           model:h3         0.772        0.785      -0.013  [-0.076, +0.051]     ties   79     True
                                   compart-focuscost:h3 model:h3         0.747        0.785      -0.038  [-0.127, +0.038]     ties   79     True
                                   model:h3             hold/focus       0.785        0.570       0.215  [+0.101, +0.316]    beats   79     True
                                   compart:h3           hold/focus       0.772        0.570       0.203  [+0.089, +0.316]    beats   79     True
                                   compart-focuscost:h3 hold/focus       0.747        0.570       0.177  [+0.063, +0.291]    beats   79     True
S4       all starts                compart:h3           model:h3         0.115        0.112       0.002  [-0.007, +0.013]     ties  400     True
                                   compart-focuscost:h3 model:h3         0.108        0.112      -0.005  [-0.030, +0.020]     ties  400     True
                                   model:h3             hold/focus       0.112        0.128      -0.015  [-0.037, +0.007]     ties  400     True
                                   compart:h3           hold/focus       0.115        0.128      -0.012  [-0.035, +0.010]     ties  400     True
                                   compart-focuscost:h3 hold/focus       0.108        0.128      -0.020  [-0.048, +0.007]     ties  400     True
         both sides started >= 80% compart:h3           model:h3         0.000        0.000       0.000  [+0.000, +0.000]     ties   15     True
                                   compart-focuscost:h3 model:h3         0.000        0.000       0.000  [+0.000, +0.000]     ties   15     True
                                   model:h3             hold/focus       0.000        0.000       0.000  [+0.000, +0.000]     ties   15     True
                                   compart:h3           hold/focus       0.000        0.000       0.000  [+0.000, +0.000]     ties   15     True
                                   compart-focuscost:h3 hold/focus       0.000        0.000       0.000  [+0.000, +0.000]     ties   15     True
S5       all starts                compart:h3           model:h3         0.522        0.530      -0.008  [-0.022, +0.007]     ties  400    False
                                   compart-focuscost:h3 model:h3         0.510        0.530      -0.020  [-0.037, -0.003]    loses  400    False
                                   model:h3             advance/focus    0.530        0.528       0.002  [-0.005, +0.010]     ties  400    False
                                   compart:h3           advance/focus    0.522        0.528      -0.005  [-0.020, +0.010]     ties  400    False
                                   compart-focuscost:h3 advance/focus    0.510        0.528      -0.018  [-0.037, +0.000]     ties  400    False
         both sides started >= 80% compart:h3           model:h3         0.541        0.553      -0.012  [-0.035, +0.000]     ties   85    False
                                   compart-focuscost:h3 model:h3         0.541        0.553      -0.012  [-0.047, +0.024]     ties   85    False
                                   model:h3             advance/focus    0.553        0.553       0.000  [+0.000, +0.000]     ties   85    False
                                   compart:h3           advance/focus    0.541        0.553      -0.012  [-0.035, +0.000]     ties   85    False
                                   compart-focuscost:h3 advance/focus    0.541        0.553      -0.012  [-0.059, +0.024]     ties   85    False
```

S1: wins per policy and planner picks

```
                          won  margin  blue_preserved
policy                                               
compart-focuscost:h3    0.680   0.088           0.284
compart:h3              0.678   0.084           0.273
fixed:advance/focus     0.672   0.077           0.263
fixed:advance/nearest   0.588   0.053           0.239
fixed:hold/focus        0.715   0.098           0.288
fixed:hold/nearest      0.625   0.066           0.253
fixed:withdraw/focus    0.000  -0.216           0.779
fixed:withdraw/nearest  0.000  -0.216           0.779
model:h3                0.675   0.080           0.269
random                  0.620   0.055           0.255
```

```
                      pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                      
compart-focuscost:h3                 0.335               0.073              0.274            0.298                  0.020     3.508    9.905
compart:h3                           0.022               0.394              0.011            0.549                  0.024     2.312   10.195
fixed:hold/focus                       NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                     NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                             0.017               0.603              0.015            0.343                  0.023     2.390    9.810
```

S2: wins per policy and planner picks

```
                          won  margin  blue_preserved
policy                                               
compart-focuscost:h3    0.662   0.081           0.275
compart:h3              0.650   0.080           0.260
fixed:advance/focus     0.655   0.076           0.241
fixed:advance/nearest   0.615   0.062           0.242
fixed:hold/focus        0.662   0.083           0.244
fixed:hold/nearest      0.615   0.061           0.238
fixed:withdraw/focus    0.000  -0.147           0.875
fixed:withdraw/nearest  0.000  -0.147           0.875
model:h3                0.658   0.084           0.260
random                  0.635   0.071           0.267
```

```
                      pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                      
compart-focuscost:h3                 0.433               0.126              0.197            0.155                  0.089     6.320   20.528
compart:h3                           0.032               0.494              0.051            0.352                  0.071     4.570   20.980
fixed:hold/focus                       NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                     NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                             0.008               0.597              0.044            0.265                  0.086     3.535   21.228
```

S3: wins per policy and planner picks

```
                          won  margin  blue_preserved
policy                                               
compart-focuscost:h3    0.745   0.194           0.623
compart:h3              0.728   0.190           0.607
fixed:advance/focus     0.535   0.014           0.246
fixed:advance/nearest   0.590   0.043           0.280
fixed:hold/focus        0.585   0.036           0.264
fixed:hold/nearest      0.570   0.046           0.282
fixed:withdraw/focus    0.000  -0.113           0.916
fixed:withdraw/nearest  0.000  -0.113           0.916
model:h3                0.735   0.200           0.640
random                  0.695   0.120           0.399
```

```
                      pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                      
compart-focuscost:h3                 0.155               0.224              0.256            0.189                  0.176     8.805   32.245
compart:h3                           0.007               0.349              0.057            0.408                  0.180     7.500   33.238
fixed:hold/focus                       NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                     NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                             0.005               0.234              0.005            0.558                  0.198     5.792   33.022
```

S4: wins per policy and planner picks

```
                          won  margin  blue_preserved
policy                                               
compart-focuscost:h3    0.108  -0.242           0.066
compart:h3              0.115  -0.262           0.062
fixed:advance/focus     0.112  -0.282           0.031
fixed:advance/nearest   0.085  -0.258           0.027
fixed:hold/focus        0.128  -0.272           0.033
fixed:hold/nearest      0.095  -0.259           0.027
fixed:withdraw/focus    0.000  -0.378           0.831
fixed:withdraw/nearest  0.000  -0.378           0.831
model:h3                0.112  -0.261           0.077
random                  0.148  -0.244           0.050
```

```
                      pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                      
compart-focuscost:h3                 0.515               0.130              0.094            0.055                  0.206     6.312   26.972
compart:h3                           0.078               0.569              0.033            0.133                  0.186     5.170   25.062
fixed:hold/focus                       NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                     NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                             0.010               0.233              0.053            0.455                  0.250     5.728   32.425
```

S5: wins per policy and planner picks

```
                          won  margin  blue_preserved
policy                                               
compart-focuscost:h3    0.510   0.049           0.327
compart:h3              0.522   0.047           0.309
fixed:advance/focus     0.528   0.023           0.263
fixed:advance/nearest   0.512   0.033           0.297
fixed:hold/focus        0.538   0.037           0.274
fixed:hold/nearest      0.520   0.038           0.300
fixed:withdraw/focus    0.000  -0.142           0.882
fixed:withdraw/nearest  0.000  -0.142           0.882
model:h3                0.530   0.043           0.302
random                  0.520   0.031           0.283
```

```
                      pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                      
compart-focuscost:h3                 0.490               0.090              0.122            0.146                  0.153     5.638   24.968
compart:h3                           0.051               0.473              0.043            0.274                  0.158     6.442   27.825
fixed:hold/focus                       NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                     NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                             0.011               0.487              0.033            0.308                  0.161     4.560   26.690
```

