# Two checks on the focus term and the score at the horizon

Script: `experiments/planning/focus_checks.py` (set-up in its docstring). Same counterfactual states as
`02_model_regret.md`. Health in ten units of the first type; unit counts in units. 95%: bootstrap over states.

## S1 (marines model)

### Check 1: focus vs nearest over the first h s

```
                                                nearest   focus  focus − nearest                 95%
h   movement quantity                                                                               
1.0 advance  Blue in reach                       4.7031  4.7792           0.0761  [+0.0400, +0.1123]
             Blue firing at own target           4.3324  3.8700          -0.4624  [-0.5139, -0.4131]
             Red health lost                     0.1046  0.0931          -0.0115  [-0.0134, -0.0098]
             Red units lost                      0.8582  1.4655           0.6073  [+0.5460, +0.6705]
             Red health lost per firing unit·s   0.0252  0.0249          -0.0003  [-0.0007, +0.0001]
    hold     Blue in reach                       4.6063  4.6801           0.0738  [+0.0409, +0.1125]
             Blue firing at own target           4.2462  3.7874          -0.4588  [-0.5058, -0.4109]
             Red health lost                     0.1015  0.0901          -0.0114  [-0.0132, -0.0096]
             Red units lost                      0.8487  1.4406           0.5920  [+0.5307, +0.6552]
             Red health lost per firing unit·s   0.0246  0.0242          -0.0003  [-0.0007, +0.0001]
2.0 advance  Blue in reach                       4.7787  4.9310           0.1523  [+0.1117, +0.1937]
             Blue firing at own target           4.4444  4.0479          -0.3964  [-0.4370, -0.3559]
             Red health lost                     0.2032  0.1868          -0.0164  [-0.0190, -0.0136]
             Red units lost                      1.9647  2.8274           0.8628  [+0.7900, +0.9418]
             Red health lost per firing unit·s   0.0229  0.0230           0.0001  [-0.0002, +0.0003]
    hold     Blue in reach                       4.6454  4.7997           0.1543  [+0.1141, +0.1957]
             Blue firing at own target           4.3208  3.9345          -0.3863  [-0.4276, -0.3440]
             Red health lost                     0.1968  0.1813          -0.0156  [-0.0182, -0.0129]
             Red units lost                      1.9253  2.7573           0.8320  [+0.7531, +0.9129]
             Red health lost per firing unit·s   0.0228  0.0230           0.0001  [-0.0001, +0.0004]
```

### Check 2: scores at h, judged by the eventual outcome

```
                                          regret (final margin)                 95%     won
score                                h                                                     
always advance/nearest               NaN                 0.0883  [+0.0784, +0.0989]  0.5766
best fixed in hindsight (hold_focus) NaN                 0.0828  [+0.0718, +0.0945]  0.5843
true best by final margin            NaN                 0.0000  [+0.0000, +0.0000]  0.7433
health margin: true at h             1.0                 0.0744  [+0.0649, +0.0845]  0.6130
health margin: model at h            1.0                 0.0839  [+0.0730, +0.0950]  0.5805
alive margin: true at h              1.0                 0.0695  [+0.0602, +0.0795]  0.6092
alive margin: model at h             1.0                 0.0852  [+0.0747, +0.0965]  0.5805
health + alive: true at h            1.0                 0.0679  [+0.0587, +0.0773]  0.6130
health + alive: model at h           1.0                 0.0852  [+0.0747, +0.0964]  0.5785
square law: true at h                1.0                 0.0766  [+0.0661, +0.0872]  0.5939
square law: model at h               1.0                 0.0850  [+0.0746, +0.0966]  0.5805
health margin: true at h             2.0                 0.0617  [+0.0525, +0.0718]  0.6245
health margin: model at h            2.0                 0.0832  [+0.0721, +0.0941]  0.5824
alive margin: true at h              2.0                 0.0683  [+0.0583, +0.0778]  0.6226
alive margin: model at h             2.0                 0.0836  [+0.0729, +0.0945]  0.5824
health + alive: true at h            2.0                 0.0687  [+0.0589, +0.0796]  0.6207
health + alive: model at h           2.0                 0.0836  [+0.0730, +0.0948]  0.5824
square law: true at h                2.0                 0.0713  [+0.0613, +0.0807]  0.6169
square law: model at h               2.0                 0.0853  [+0.0748, +0.0959]  0.5805
health margin: true at h             3.0                 0.0536  [+0.0444, +0.0634]  0.6571
health margin: model at h            3.0                 0.0822  [+0.0717, +0.0931]  0.5862
alive margin: true at h              3.0                 0.0599  [+0.0509, +0.0692]  0.6341
alive margin: model at h             3.0                 0.0820  [+0.0714, +0.0934]  0.5862
health + alive: true at h            3.0                 0.0568  [+0.0482, +0.0667]  0.6379
health + alive: model at h           3.0                 0.0821  [+0.0717, +0.0930]  0.5862
square law: true at h                3.0                 0.0655  [+0.0564, +0.0750]  0.6303
square law: model at h               3.0                 0.0853  [+0.0746, +0.0963]  0.5805
health margin: true at h             4.0                 0.0465  [+0.0380, +0.0560]  0.6686
health margin: model at h            4.0                 0.0828  [+0.0722, +0.0932]  0.5881
alive margin: true at h              4.0                 0.0506  [+0.0426, +0.0586]  0.6571
alive margin: model at h             4.0                 0.0802  [+0.0698, +0.0907]  0.5920
health + alive: true at h            4.0                 0.0485  [+0.0402, +0.0574]  0.6609
health + alive: model at h           4.0                 0.0827  [+0.0723, +0.0937]  0.5862
square law: true at h                4.0                 0.0595  [+0.0510, +0.0687]  0.6341
square law: model at h               4.0                 0.0852  [+0.0742, +0.0962]  0.5805
```

## S2 (mixed model)

### Check 1: focus vs nearest over the first h s

```
                                                nearest   focus  focus − nearest                 95%
h   movement quantity                                                                               
1.0 advance  Blue in reach                       4.7101  4.7249           0.0148  [-0.0092, +0.0393]
             Blue firing at own target           4.3472  4.0895          -0.2577  [-0.2943, -0.2209]
             Red health lost                     0.1189  0.1054          -0.0135  [-0.0155, -0.0115]
             Red units lost                      0.5368  0.9950           0.4582  [+0.4080, +0.5067]
             Red health lost per firing unit·s   0.0290  0.0272          -0.0018  [-0.0024, -0.0013]
    hold     Blue in reach                       4.6371  4.6501           0.0130  [-0.0100, +0.0349]
             Blue firing at own target           4.2893  4.0309          -0.2584  [-0.2950, -0.2218]
             Red health lost                     0.1162  0.1025          -0.0136  [-0.0155, -0.0117]
             Red units lost                      0.5301  0.9900           0.4599  [+0.4080, +0.5100]
             Red health lost per firing unit·s   0.0287  0.0267          -0.0019  [-0.0024, -0.0014]
2.0 advance  Blue in reach                       4.8438  4.8997           0.0560  [+0.0260, +0.0857]
             Blue firing at own target           4.5339  4.3164          -0.2175  [-0.2505, -0.1824]
             Red health lost                     0.2338  0.2120          -0.0218  [-0.0250, -0.0189]
             Red units lost                      1.3003  2.0188           0.7184  [+0.6604, +0.7816]
             Red health lost per firing unit·s   0.0263  0.0250          -0.0013  [-0.0016, -0.0009]
    hold     Blue in reach                       4.7426  4.7988           0.0562  [+0.0255, +0.0878]
             Blue firing at own target           4.4439  4.2330          -0.2109  [-0.2435, -0.1746]
             Red health lost                     0.2281  0.2066          -0.0215  [-0.0248, -0.0185]
             Red units lost                      1.2696  1.9727           0.7031  [+0.6399, +0.7679]
             Red health lost per firing unit·s   0.0262  0.0248          -0.0014  [-0.0017, -0.0010]
```

### Check 2: scores at h, judged by the eventual outcome

```
                                          regret (final margin)                 95%     won
score                                h                                                     
always advance/nearest               NaN                 0.1682  [+0.1496, +0.1863]  0.5970
best fixed in hindsight (hold_focus) NaN                 0.1577  [+0.1379, +0.1766]  0.6120
true best by final margin            NaN                 0.0000  [+0.0000, +0.0000]  0.7659
health margin: true at h             1.0                 0.1516  [+0.1345, +0.1693]  0.6120
health margin: model at h            1.0                 0.1581  [+0.1399, +0.1781]  0.6087
alive margin: true at h              1.0                 0.1526  [+0.1348, +0.1713]  0.6221
alive margin: model at h             1.0                 0.1555  [+0.1372, +0.1750]  0.6104
health + alive: true at h            1.0                 0.1471  [+0.1297, +0.1663]  0.6237
health + alive: model at h           1.0                 0.1559  [+0.1369, +0.1757]  0.6104
square law: true at h                1.0                 0.1589  [+0.1408, +0.1773]  0.6137
square law: model at h               1.0                 0.1526  [+0.1348, +0.1708]  0.6137
health margin: true at h             2.0                 0.1318  [+0.1146, +0.1487]  0.6288
health margin: model at h            2.0                 0.1543  [+0.1357, +0.1752]  0.6120
alive margin: true at h              2.0                 0.1384  [+0.1206, +0.1558]  0.6321
alive margin: model at h             2.0                 0.1566  [+0.1379, +0.1762]  0.6104
health + alive: true at h            2.0                 0.1265  [+0.1106, +0.1443]  0.6388
health + alive: model at h           2.0                 0.1560  [+0.1377, +0.1746]  0.6104
square law: true at h                2.0                 0.1405  [+0.1242, +0.1576]  0.6254
square law: model at h               2.0                 0.1556  [+0.1374, +0.1746]  0.6104
health margin: true at h             3.0                 0.1214  [+0.1042, +0.1392]  0.6405
health margin: model at h            3.0                 0.1538  [+0.1354, +0.1733]  0.6137
alive margin: true at h              3.0                 0.1314  [+0.1149, +0.1491]  0.6388
alive margin: model at h             3.0                 0.1563  [+0.1377, +0.1756]  0.6070
health + alive: true at h            3.0                 0.1231  [+0.1058, +0.1406]  0.6455
health + alive: model at h           3.0                 0.1534  [+0.1345, +0.1728]  0.6120
square law: true at h                3.0                 0.1296  [+0.1132, +0.1463]  0.6388
square law: model at h               3.0                 0.1584  [+0.1410, +0.1781]  0.6070
health margin: true at h             4.0                 0.1105  [+0.0937, +0.1267]  0.6572
health margin: model at h            4.0                 0.1525  [+0.1336, +0.1721]  0.6154
alive margin: true at h              4.0                 0.1252  [+0.1085, +0.1423]  0.6488
alive margin: model at h             4.0                 0.1545  [+0.1357, +0.1734]  0.6104
health + alive: true at h            4.0                 0.1121  [+0.0957, +0.1290]  0.6639
health + alive: model at h           4.0                 0.1523  [+0.1330, +0.1719]  0.6104
square law: true at h                4.0                 0.1295  [+0.1139, +0.1460]  0.6405
square law: model at h               4.0                 0.1593  [+0.1416, +0.1782]  0.6070
```
