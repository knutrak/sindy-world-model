# The health-spread state: results

Plan: `09_spread_state_plan.md` (pre-registered 21 Nov). Script: `experiments/planning/spread_state.py`.
Spread tables: `scripts/build_spread_states.py`. Fitted on train only; tested on val, the counterfactual
sets and 400 fresh seeds per scenario (15 x00 000 +).

## Choice of the measure and H1 (train, 5-fold CV by battle)

```
                                CV R² frozen form  CV R² with spread
measure scenario side type                                          
even    S1       blue marine               0.1991             0.2001
                 red  marine               0.3138             0.3138
        S2       blue marine               0.1267             0.1271
                      marauder             0.0477             0.0780
                 red  marine               0.1924             0.1924
                      marauder             0.0870             0.1220
fragile S1       blue marine               0.1991             0.2001
                 red  marine               0.3138             0.3138
        S2       blue marine               0.1267             0.1271
                      marauder             0.0477             0.0548
                 red  marine               0.1924             0.2065
                      marauder             0.0870             0.0943
```

Pooled CV R²: {"even": {"frozen form": 0.1959, "with spread": 0.2011}, "fragile": {"frozen form": 0.1959, "with spread": 0.1999}}.
Chosen: **even** (rule: larger pooled R², ties within 0.005 to evenness).

## The fitted spread variant

```
S1 (marines model)
p_B^marine(s,u) = 1·(+0.6647 +0.0156·focus -0.0045·hold -0.0897·wd)      [frozen: 0.662 + 0.000·focus]
p_R^marine(s,u) = 1·(+0.5618 +0.2600·focus +0.0012·hold)      [frozen: 0.562 + 0.260·focus]
s_B^marine' = 1·(-0.0964 -0.0020·focus -0.0061·hold +0.0651·wd +0.0319·focus*far +0.1832·hold*far) + s·(+0.0848 +0.0064·focus +0.0046·hold -0.0519·wd -0.0326·focus*far -0.1714·hold*far) + r·(+0.3997 +0.0580·focus -0.0209·hold -0.2165·wd +4.1003·focus*far -2.3097·hold*far) + r*s·(-0.4678 -0.0757·focus +0.0327·hold +0.1754·wd -5.0696·focus*far +1.6863·hold*far)
s_R^marine' = 1·(-0.1246 +0.1063·focus -0.0126·hold +0.0594·wd -0.1020·focus*far +0.2018·hold*far) + s·(+0.1029 -0.0903·focus +0.0129·hold -0.0436·wd +0.0921·focus*far -0.1841·hold*far) + r·(+0.3323 +0.0724·focus +0.0081·hold -3.5190·focus*far -2.7085·hold*far) + r*s·(-0.3731 -0.1063·focus -0.0015·hold +2.1695·focus*far +1.6958·hold*far)

S2 (mixed model)
p_B^marine(s,u) = 1·(+0.5672 +0.0229·focus +0.0104·hold -0.0638·wd)      [frozen: 0.578 + 0.000·focus]
p_B^marauder(s,u) = 1·(+0.1941 +0.0164·focus +0.0038·hold -0.0301·wd) + (1-s)·(+1.5216 +0.0102·focus +0.0350·hold +0.5005·wd)      [frozen: 0.444 + 0.000·focus]
p_R^marine(s,u) = 1·(+0.5047 +0.1554·focus +0.0026·hold)      [frozen: 0.506 + 0.155·focus]
p_R^marauder(s,u) = 1·(+0.1594 +0.0881·focus -0.0009·hold) + (1-s)·(+1.2555 +0.3042·focus +0.0203·hold)      [frozen: 0.376 + 0.139·focus]
s_B^marine' = 1·(-0.0451 -0.0035·focus -0.0156·hold +0.0238·wd +0.0253·focus*far +0.0559·hold*far) + s·(+0.0407 +0.0075·focus +0.0193·hold -0.0166·wd -0.0292·focus*far -0.0554·hold*far) + r·(+0.4570 +0.0419·focus +0.0200·hold -0.2719·wd +1.5395·focus*far +2.2375·hold*far) + r*s·(-0.6015 -0.0637·focus -0.0205·hold +0.2637·wd -1.9258·focus*far -2.9825·hold*far)
s_B^marauder' = 1·(-0.0637 -0.0163·focus -0.0158·hold +0.0432·wd +0.0338·focus*far +0.0766·hold*far) + s·(+0.0691 +0.0202·focus +0.0170·hold -0.0453·wd -0.0396·focus*far -0.0818·hold*far) + r·(+0.8428 +0.0787·focus +0.0541·hold -0.0583·wd +0.9879·focus*far +2.3690·hold*far) + r*s·(-1.1535 -0.1188·focus -0.0418·hold -0.0285·wd -1.2748·focus*far -2.9395·hold*far)
s_R^marine' = 1·(-0.0406 +0.0192·focus -0.0302·hold +0.0164·wd +0.0172·focus*far +0.0665·hold*far) + s·(+0.0319 -0.0040·focus +0.0347·hold -0.0119·wd -0.0328·focus*far -0.0653·hold*far) + r·(+0.4071 +0.1023·focus +0.0273·hold +0.9696·focus*far +0.5978·hold*far) + r*s·(-0.5388 -0.1349·focus -0.0368·hold -1.9438·focus*far -1.2675·hold*far)
s_R^marauder' = 1·(-0.0896 +0.0162·focus -0.0187·hold +0.0632·wd +0.0164·focus*far +0.0960·hold*far) + s·(+0.0920 -0.0118·focus +0.0195·hold -0.0661·wd -0.0214·focus*far -0.0987·hold*far) + r·(+0.7158 +0.3350·focus +0.0259·hold +1.1021·focus*far +1.1639·hold*far) + r*s·(-0.9669 -0.4136·focus -0.0096·hold -1.5571·focus*far -1.6180·hold*far)

```

Spread dynamics, CV R² (train):

```
                        s' CV R²  median s
scenario side type                        
S1       blue marine      0.0092    0.9188
         red  marine      0.0180    0.9331
S2       blue marine      0.0150    0.9373
              marauder    0.0339    0.9559
         red  marine      0.0269    0.9331
              marauder    0.0420    0.9631
```

## H2: Red's focus term at three levels of evenness

```
                        focus − nearest in p  frozen focus term
scenario type     s                                            
S1       marine   1.00                0.2600             0.2600
                  0.85                0.2600             0.2600
                  0.70                0.2600             0.2600
S2       marine   1.00                0.1554             0.1554
                  0.85                0.1554             0.1554
                  0.70                0.1554             0.1554
         marauder 1.00                0.0881             0.1395
                  0.85                0.1337             0.1395
                  0.70                0.1793             0.1395
```

## H3: focus terms in Red's evenness dynamics

```
{"scenario": "S1", "type": "marine", "focus terms in Red's s'": {"1|focus": 0.10625080998906748, "1|focus*far": -0.10201739452094197, "s|focus": -0.09033812481483358, "s|focus*far": 0.09209690463425939, "r|focus": 0.07243568019509455, "r|focus*far": -3.519008718247375, "r*s|focus": -0.10627284901331797, "r*s|focus*far": 2.1694815982415574}}
{"scenario": "S2", "type": "marine", "focus terms in Red's s'": {"1|focus": 0.01916899544284788, "1|focus*far": 0.017224491572955202, "s|focus": -0.004024315610578823, "s|focus*far": -0.03275000851372717, "r|focus": 0.10229580655726078, "r|focus*far": 0.9696135444417433, "r*s|focus": -0.13491086918386427, "r*s|focus*far": -1.9437978836250307}}
{"scenario": "S2", "type": "marauder", "focus terms in Red's s'": {"1|focus": 0.0162128323021479, "1|focus*far": 0.016421182798592317, "s|focus": -0.01181157914810463, "s|focus*far": -0.02135087433580634, "r|focus": 0.33500604791461747, "r|focus*far": 1.1020870900436066, "r*s|focus": -0.413565346580917, "r*s|focus*far": -1.5571106743661782}}
```

## H4: alive error on val (team, units; paired by start point, bootstrap by battle)

```
                                  alive error, frozen (units)  alive error  difference                 95%  health error difference          health 95%
scenario h   model                                                                                                                                     
S1       1.0 spread                                    0.8392       0.8382     -0.0010  [-0.0016, -0.0004]                  -0.0000  [-0.0000, -0.0000]
             spread + focus cost                       0.8392       0.8412      0.0019  [+0.0007, +0.0032]                  -0.0001  [-0.0001, -0.0000]
         2.0 spread                                    1.1010       1.0989     -0.0022  [-0.0035, -0.0009]                  -0.0001  [-0.0001, -0.0001]
             spread + focus cost                       1.1010       1.1026      0.0015  [-0.0016, +0.0048]                  -0.0001  [-0.0003, +0.0001]
         4.0 spread                                    1.8261       1.8147     -0.0114  [-0.0141, -0.0087]                  -0.0003  [-0.0004, -0.0003]
             spread + focus cost                       1.8261       1.8095     -0.0166  [-0.0264, -0.0074]                  -0.0008  [-0.0012, -0.0003]
S2       1.0 spread                                    0.7574       0.7377     -0.0196  [-0.0210, -0.0183]                  -0.0001  [-0.0001, -0.0001]
             spread + focus cost                       0.7574       0.7390     -0.0184  [-0.0197, -0.0169]                  -0.0002  [-0.0002, -0.0002]
         2.0 spread                                    1.0028       0.9847     -0.0180  [-0.0208, -0.0152]                  -0.0002  [-0.0002, -0.0001]
             spread + focus cost                       1.0028       0.9874     -0.0154  [-0.0184, -0.0124]                  -0.0003  [-0.0004, -0.0002]
         4.0 spread                                    1.4777       1.4579     -0.0197  [-0.0259, -0.0140]                  -0.0002  [-0.0004, +0.0000]
             spread + focus cost                       1.4777       1.4622     -0.0155  [-0.0217, -0.0094]                  -0.0001  [-0.0004, +0.0002]
```

## H5: ranking on the counterfactual sets

```
                                  regret  rank corr  always adv/nearest  picks focus  eventual won  eventual regret
scenario h   model                                                                                                 
S1       1.0 frozen               0.0170     0.2386              0.0052       0.9847        0.5805           0.0839
             spread               0.0169     0.2391              0.0052       0.9808        0.5824           0.0837
             spread + focus cost  0.0055     0.5496              0.0052       0.0038        0.5747           0.0872
         2.0 frozen               0.0225     0.3477              0.0109       0.9636        0.5824           0.0832
             spread               0.0225     0.3496              0.0109       0.9598        0.5824           0.0832
             spread + focus cost  0.0109     0.5296              0.0109       0.0460        0.5728           0.0871
         4.0 frozen               0.0308     0.3873              0.0267       0.9195        0.5881           0.0828
             spread               0.0308     0.3859              0.0267       0.9119        0.5862           0.0828
             spread + focus cost  0.0256     0.4263              0.0267       0.2280        0.5785           0.0840
         8.0 frozen               0.0580     0.1800              0.0608       0.9272        0.5785           0.0855
             spread               0.0580     0.1759              0.0608       0.9176        0.5785           0.0855
             spread + focus cost  0.0606     0.1500              0.0608       0.6073        0.5843           0.0876
S2       1.0 frozen               0.0188     0.2971              0.0063       0.9716        0.6087           0.1581
             spread               0.0187     0.3005              0.0063       0.9649        0.6087           0.1588
             spread + focus cost  0.0063     0.5695              0.0063       0.0033        0.6003           0.1673
         2.0 frozen               0.0297     0.3240              0.0121       0.9599        0.6120           0.1543
             spread               0.0298     0.3218              0.0121       0.9599        0.6120           0.1544
             spread + focus cost  0.0122     0.5827              0.0121       0.0067        0.6003           0.1651
         4.0 frozen               0.0444     0.3477              0.0355       0.8930        0.6154           0.1525
             spread               0.0443     0.3463              0.0355       0.8846        0.6154           0.1511
             spread + focus cost  0.0377     0.4613              0.0355       0.0936        0.6087           0.1592
         8.0 frozen               0.0747     0.2151              0.0884       0.8261        0.6288           0.1511
             spread               0.0746     0.2086              0.0884       0.8127        0.6271           0.1500
             spread + focus cost  0.0816     0.1862              0.0884       0.5569        0.6087           0.1556
```

```
                                  regret − frozen                 95%
scenario h   model                                                   
S1       1.0 spread                       -0.0001  [-0.0002, +0.0000]
             spread + focus cost          -0.0115  [-0.0136, -0.0094]
         2.0 spread                        0.0000  [-0.0001, +0.0001]
             spread + focus cost          -0.0116  [-0.0148, -0.0083]
         4.0 spread                        0.0000  [-0.0005, +0.0005]
             spread + focus cost          -0.0051  [-0.0100, -0.0004]
         8.0 spread                       -0.0000  [-0.0003, +0.0002]
             spread + focus cost           0.0027  [-0.0009, +0.0062]
S2       1.0 spread                       -0.0001  [-0.0003, +0.0000]
             spread + focus cost          -0.0126  [-0.0147, -0.0104]
         2.0 spread                        0.0001  [+0.0000, +0.0002]
             spread + focus cost          -0.0175  [-0.0212, -0.0137]
         4.0 spread                       -0.0001  [-0.0012, +0.0010]
             spread + focus cost          -0.0067  [-0.0132, -0.0004]
         8.0 spread                       -0.0002  [-0.0021, +0.0019]
             spread + focus cost           0.0069  [+0.0009, +0.0128]
```

## H6: closed loop on fresh seeds (wins, paired over 400 seeds)

```
                                            arm won  against won  difference               95% read-out    n  flagged  same outcome as frozen planner
scenario arm                 against                                                                                                                 
S1       spread:h3           model:h3         0.622        0.622       0.000  [+0.000, +0.000]     ties  400    False                           1.000
         spread-focuscost:h3 model:h3         0.635        0.622       0.012  [-0.028, +0.052]     ties  400    False                             NaN
         model:h3            hold/focus       0.622        0.645      -0.022  [-0.055, +0.010]     ties  400    False                             NaN
         spread:h3           hold/focus       0.622        0.645      -0.022  [-0.055, +0.010]     ties  400    False                             NaN
         spread-focuscost:h3 hold/focus       0.635        0.645      -0.010  [-0.052, +0.030]     ties  400    False                             NaN
S2       spread:h3           model:h3         0.622        0.620       0.002  [+0.000, +0.007]     ties  400    False                           0.998
         spread-focuscost:h3 model:h3         0.625        0.620       0.005  [-0.040, +0.050]     ties  400    False                             NaN
         model:h3            advance/focus    0.620        0.630      -0.010  [-0.028, +0.007]     ties  400    False                             NaN
         spread:h3           advance/focus    0.622        0.630      -0.008  [-0.025, +0.007]     ties  400    False                             NaN
         spread-focuscost:h3 advance/focus    0.625        0.630      -0.005  [-0.052, +0.040]     ties  400    False                             NaN
S3       spread:h3           model:h3         0.738        0.718       0.020  [+0.000, +0.043]     ties  400     True                           0.955
         spread-focuscost:h3 model:h3         0.732        0.718       0.015  [-0.020, +0.050]     ties  400     True                             NaN
         model:h3            hold/focus       0.718        0.582       0.135  [+0.083, +0.188]    beats  400     True                             NaN
         spread:h3           hold/focus       0.738        0.582       0.155  [+0.102, +0.207]    beats  400     True                             NaN
         spread-focuscost:h3 hold/focus       0.732        0.582       0.150  [+0.098, +0.205]    beats  400     True                             NaN
S4       spread:h3           model:h3         0.065        0.065       0.000  [+0.000, +0.000]     ties  400     True                           1.000
         spread-focuscost:h3 model:h3         0.075        0.065       0.010  [-0.020, +0.043]     ties  400     True                             NaN
         model:h3            hold/focus       0.065        0.062       0.002  [-0.015, +0.020]     ties  400     True                             NaN
         spread:h3           hold/focus       0.065        0.062       0.002  [-0.015, +0.020]     ties  400     True                             NaN
         spread-focuscost:h3 hold/focus       0.075        0.062       0.012  [-0.018, +0.045]     ties  400     True                             NaN
S5       spread:h3           model:h3         0.498        0.495       0.002  [+0.000, +0.007]     ties  400    False                           0.998
         spread-focuscost:h3 model:h3         0.495        0.495       0.000  [-0.020, +0.020]     ties  400    False                             NaN
         model:h3            advance/focus    0.495        0.500      -0.005  [-0.015, +0.005]     ties  400    False                             NaN
         spread:h3           advance/focus    0.498        0.500      -0.002  [-0.010, +0.005]     ties  400    False                             NaN
         spread-focuscost:h3 advance/focus    0.495        0.500      -0.005  [-0.022, +0.013]     ties  400    False                             NaN
```

S1 wins per policy and planner picks:

```
                          won  margin  blue_preserved
policy                                               
fixed:advance/focus     0.618   0.060           0.245
fixed:advance/nearest   0.592   0.039           0.223
fixed:hold/focus        0.645   0.066           0.245
fixed:hold/nearest      0.612   0.050           0.230
fixed:withdraw/focus    0.000  -0.224           0.771
fixed:withdraw/nearest  0.000  -0.224           0.771
model:h3                0.622   0.063           0.252
random                  0.570   0.036           0.233
spread-focuscost:h3     0.635   0.061           0.256
spread:h3               0.622   0.063           0.253
```

```
                     pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                     
fixed:hold/focus                      NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                    NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                            0.014               0.577              0.024            0.354                  0.031     2.425   10.952
spread-focuscost:h3                 0.118               0.026              0.541            0.291                  0.025     2.968    9.768
spread:h3                           0.009               0.147              0.055            0.758                  0.031     2.138   11.050
```

S2 wins per policy and planner picks:

```
                          won  margin  blue_preserved
policy                                               
fixed:advance/focus     0.630   0.054           0.229
fixed:advance/nearest   0.565   0.040           0.221
fixed:hold/focus        0.632   0.058           0.234
fixed:hold/nearest      0.605   0.059           0.242
fixed:withdraw/focus    0.000  -0.159           0.874
fixed:withdraw/nearest  0.000  -0.159           0.874
model:h3                0.620   0.063           0.253
random                  0.612   0.062           0.261
spread-focuscost:h3     0.625   0.064           0.269
spread:h3               0.622   0.063           0.252
```

```
                     pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                     
fixed:hold/focus                      NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                    NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                            0.010               0.560              0.058            0.258                  0.114     4.515   25.415
spread-focuscost:h3                 0.498               0.187              0.166            0.034                  0.115     5.908   25.602
spread:h3                           0.026               0.712              0.064            0.085                  0.112     4.075   24.990
```

S3 wins per policy and planner picks:

```
                          won  margin  blue_preserved
policy                                               
fixed:advance/focus     0.532   0.010           0.243
fixed:advance/nearest   0.510   0.022           0.257
fixed:hold/focus        0.582   0.033           0.272
fixed:hold/nearest      0.528   0.036           0.272
fixed:withdraw/focus    0.000  -0.107           0.925
fixed:withdraw/nearest  0.000  -0.107           0.925
model:h3                0.718   0.207           0.656
random                  0.685   0.101           0.376
spread-focuscost:h3     0.732   0.200           0.664
spread:h3               0.738   0.210           0.640
```

```
                     pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                     
fixed:hold/focus                      NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                    NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                            0.007               0.230              0.007            0.579                  0.178     5.988   34.028
spread-focuscost:h3                 0.077               0.088              0.423            0.235                  0.176     7.152   33.272
spread:h3                           0.009               0.181              0.058            0.587                  0.165     6.818   32.275
```

S4 wins per policy and planner picks:

```
                          won  margin  blue_preserved
policy                                               
fixed:advance/focus     0.062  -0.307           0.015
fixed:advance/nearest   0.055  -0.273           0.014
fixed:hold/focus        0.062  -0.299           0.017
fixed:hold/nearest      0.055  -0.269           0.014
fixed:withdraw/focus    0.000  -0.367           0.838
fixed:withdraw/nearest  0.000  -0.367           0.838
model:h3                0.065  -0.290           0.060
random                  0.115  -0.267           0.040
spread-focuscost:h3     0.075  -0.252           0.058
spread:h3               0.065  -0.286           0.062
```

```
                     pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                     
fixed:hold/focus                      NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                    NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                            0.013               0.198              0.050            0.487                  0.252     5.825   32.242
spread-focuscost:h3                 0.406               0.035              0.282            0.020                  0.258     7.068   31.070
spread:h3                           0.097               0.399              0.089            0.157                  0.259     6.500   32.792
```

S5 wins per policy and planner picks:

```
                          won  margin  blue_preserved
policy                                               
fixed:advance/focus     0.500  -0.005           0.251
fixed:advance/nearest   0.490   0.002           0.270
fixed:hold/focus        0.505  -0.000           0.253
fixed:hold/nearest      0.482  -0.001           0.265
fixed:withdraw/focus    0.000  -0.165           0.859
fixed:withdraw/nearest  0.000  -0.165           0.859
model:h3                0.495   0.006           0.281
random                  0.520   0.019           0.284
spread-focuscost:h3     0.495   0.020           0.306
spread:h3               0.498   0.006           0.279
```

```
                     pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  switches  replans
policy                                                                                                                                     
fixed:hold/focus                      NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
fixed:hold/nearest                    NaN                 NaN                NaN              NaN                    NaN       NaN      NaN
model:h3                            0.011               0.456              0.027            0.322                  0.184     4.408   27.695
spread-focuscost:h3                 0.486               0.157              0.184            0.026                  0.147     5.165   23.468
spread:h3                           0.059               0.624              0.042            0.106                  0.168     4.838   26.110
```

