# exp08 pilot: 300 mixed marine / marauder battles (19 Nov)

Script: `experiments/mixed/pilot.py`. Exploration only.

## Battle length and Blue win rate by marauder share

```
           battles  duration_s  blue_win
share_bin                               
~1/4            92      15.573     0.641
~1/3           111      15.134     0.604
~1/2            97      17.285     0.557
```

## Survival on the winning side (share of starting units alive at the end)

```
          survive
type             
marauder    0.561
marine      0.266
```

## Fire allocation

Share of each attacker type's fire (units with a target in range) that is on marauders, against
the marauder share among the enemy's alive units at the same steps. Blue's rule: nearest or
focus (weakest by absolute hp, so marines); Red's AI: nearest new target, keeps its target.

```
   side  attacker  fire on marauders (measured)  marauder share of enemies alive (same steps, fire-weighted)
0  blue    marine                         0.532                                                        0.541
1  blue  marauder                         0.565                                                        0.601
2   red    marine                         0.507                                                        0.401
3   red  marauder                         0.527                                                        0.466
```

## Preview: damage matrix from measured fire (one-step least squares, pilot only)

Each type's health rate (SG 9) regressed on the measured number of attackers of each type that
are shooting a unit of that type (in range). Units: target units removed per second per firing
attacker. Game value: (damage + bonus − armor) / cooldown rounded up to ticks / target hp.
This is a preview of checkpoint 2 with *measured* allocation; it says nothing yet about closures.

```
   attacker (side)    target  fitted   game  fitted / game
0     marine (red)    marine   0.213  0.213          1.000
1   marauder (red)    marine   0.171  0.198          0.868
2     marine (red)  marauder   0.065  0.064          1.014
3   marauder (red)  marauder   0.121  0.135          0.896
4    marine (blue)    marine   0.195  0.213          0.916
5  marauder (blue)    marine   0.167  0.198          0.847
6    marine (blue)  marauder   0.063  0.064          0.990
7  marauder (blue)  marauder   0.116  0.135          0.861
```
