# The oracle with the planners' settings, and exploitation: results

Plan: `38_oracle_setup_plan.md` (approved 22 Nov). Script: `experiments/planning/oracle38.py`.
Seeds: `34`'s (21 x00 000 +), paired with every policy played there.

## Wins

```
                        S2     S4     S5
policy                                  
fixed:advance/focus  0.635    NaN  0.555
fixed:kite/focus     0.855  0.242  0.602
fixed:kite/nearest   0.895  0.290  0.615
mlp:h3               0.348  0.098  0.258
model:h3             0.628  0.070  0.560
musigma:h3           0.800  0.200  0.575
oracle7-h3           0.975  0.560  0.710
oracle7-h8           0.982  0.750  0.820
fixed:hold/focus       NaN  0.070    NaN
```

## Paired comparisons

```
scenario rule     policy      against  difference              95% read-out   n
      S2   O1 oracle7-h3 kite/nearest       0.080 [+0.052, +0.110]    beats 400
      S2   O2 oracle7-h8 kite/nearest       0.088 [+0.060, +0.117]    beats 400
      S2   O3 oracle7-h3   musigma:h3       0.175 [+0.140, +0.212]    beats 400
      S4   O1 oracle7-h3 kite/nearest       0.270 [+0.223, +0.318]    beats 400
      S4   O2 oracle7-h8 kite/nearest       0.460 [+0.407, +0.510]    beats 400
      S4   O3 oracle7-h3   musigma:h3       0.360 [+0.312, +0.410]    beats 400
      S5   O1 oracle7-h3 kite/nearest       0.095 [+0.068, +0.125]    beats 400
      S5   O2 oracle7-h8 kite/nearest       0.205 [+0.165, +0.247]    beats 400
      S5   O3 oracle7-h3   musigma:h3       0.135 [+0.102, +0.170]    beats 400
```

**Interpretation fixed in the plan: the setup claim is wrong for these settings: a perfect model would beat fixed kite, so model accuracy matters; 37's wording and the supervisor doc are to be corrected.**

## Exploitation check (100 seeds per scenario; margin gain over the next 1 s, team health units)

```
scenario    planner  re-plans  optimism (pred − real)                95%  predicted uplift (chosen − mean)         uplift 95%
      S2     mlp:h3      5040                  0.0059 [+0.0039, +0.0083]                           -0.0048 [-0.0075, -0.0019]
      S2   model:h3      2434                  0.0039 [+0.0027, +0.0054]                            0.0070 [+0.0053, +0.0094]
      S2 musigma:h3      1628                 -0.0025 [-0.0052, -0.0005]                            0.0113 [+0.0092, +0.0143]
      S4     mlp:h3      2107                  0.0056 [+0.0040, +0.0081]                            0.0043 [+0.0014, +0.0083]
      S4   model:h3      3425                  0.0001 [-0.0007, +0.0009]                            0.0027 [+0.0020, +0.0037]
      S4 musigma:h3      1801                 -0.0010 [-0.0025, +0.0003]                            0.0089 [+0.0074, +0.0110]
      S5     mlp:h3      4145                  0.0121 [+0.0078, +0.0171]                           -0.0056 [-0.0102, -0.0009]
      S5   model:h3      2691                  0.0043 [+0.0028, +0.0064]                            0.0063 [+0.0049, +0.0082]
      S5 musigma:h3      1269                 -0.0016 [-0.0053, +0.0013]                            0.0134 [+0.0113, +0.0161]
```

## Read-outs against the plan (`38`)

| | rule | result |
|---|---|---|
| **O1** | exact oracle, the planners' settings (7 candidates, 3 s horizon) − fixed kite | **beats in all three**: S2 +8.0 [+5.2, +11.0], S4 +27.0 [+22.3, +31.8], S5 +9.5 [+6.8, +12.5] |
| O2 | the same with an 8 s horizon | beats in all three: S2 +8.8, S4 +46.0, S5 +20.5 |
| O3 | 3 s oracle − μ, σ planner | +17.5, +36.0, +13.5 |
| E | optimism of the chosen intent (predicted − realised 1 s margin gain) | MLP +0.006 to +0.012 (intervals above 0 in all three); μ, σ −0.001 to −0.003; frozen 0.000 to +0.004. The MLP's intervals do not overlap the μ, σ planner's in any scenario |

**Interpretation fixed in the plan:** the setup claim is wrong for these settings. A perfect model with the
planners' own settings (3 s horizon, 7 candidates, re-plan 1 s, commit 2 s) beats fixed kite in S2, S4 and S5, so
model accuracy matters. `37`'s wording is corrected.

Facts beside the rules:
- A longer horizon helps further where the battles are long and uneven (S4: 56% at 3 s, 75% at 8 s; S5: 71% → 82%).
- Exploitation is measured: the MLP planner's chosen intent gains less than it predicts, by 0.006-0.012 team
  health units per second (about 1-3% of a side's full health per second of play). The μ, σ planner is not
  optimistic about its choices.
- The world-model planners' remaining gap to a perfect model at the same settings is 13.5-36 win points (O3).
