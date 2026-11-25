# Exploratory: stalkers alone, and one model for several scenarios (val only)

## Stalkers alone (exp07 train 1500, val 300; reach capped at 6)

```
H_B^stalker' = -0.0679·R[stalker]*s[stalker]
H_R^stalker' = -0.0693·R[stalker]*s[stalker]
(ln A_B^stalker)' = (0.657) · (ln H_B^stalker)'
(ln A_R^stalker)' = (0.545 +0.215·focus) · (ln H_R^stalker)'
d' = -0.246·walk_r*wd +1.090·danger*wd -3.965·close*adv -4.381·close*hold*far -3.760·close*hold*contact
```

```
                          fitted   game  fitted / game
term                                                  
stalker (red) → stalker    0.068  0.079          0.854
stalker (blue) → stalker   0.069  0.079          0.872
```

## One model, several scenarios (health in ten marines' hp)

```
                val set                    model  health_1s  health_2s  health_4s  margin_r2_2s  margin_r2_4s  alive_4s  distance_2s  horizon
0               marines              persistence     0.0622     0.1232     0.2334       -0.0214       -0.0278    2.9544       0.5857      NaN
1               marines                own model     0.0164     0.0277     0.0523        0.4012        0.0798    0.9130       0.3594      2.0
2               marines  triple model, unchanged     0.0174     0.0330     0.0645        0.4274        0.2155    1.0921       0.4338      3.0
3             marauders              persistence     0.1003     0.1992     0.3883       -0.0027       -0.0052    1.7735       0.7296      NaN
4             marauders                own model     0.0327     0.0565     0.1072        0.3436        0.0907    0.7254       0.4666      2.0
5             marauders  triple model, unchanged     0.0332     0.0568     0.1102        0.3748        0.1665    0.8247       0.4676      2.0
6              stalkers              persistence     0.0748     0.1457     0.2858       -0.0303       -0.0457    0.9747       0.6895      NaN
7              stalkers                own model     0.0349     0.0514     0.0861        0.3911        0.2929    0.5398       0.4266      3.0
8              stalkers  triple model, unchanged     0.0354     0.0560     0.1021        0.3657        0.2233    0.5816       0.5144      3.0
9   marines + marauders              persistence     0.0641     0.1267     0.2451       -0.0330       -0.0504    1.9288       0.6511      NaN
10  marines + marauders                own model     0.0200     0.0346     0.0640        0.3743        0.1476    0.7388       0.4318      2.0
11  marines + marauders  triple model, unchanged     0.0202     0.0356     0.0678        0.3798        0.1546    0.7832       0.4203      2.0
```
