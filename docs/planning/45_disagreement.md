# Where the model planners disagree with a perfect model: results

Plan: `44_disagreement_plan.md` (exploratory, read-outs fixed before the run).
Script: `experiments/planning/disagree44.py`. Oracle `oracle5-h3` replays, 100 seeds per scenario;
agreement merges hold/advance twins in contact; regret in hp under the oracle's true 3 s scores.

```
            by                group  re-plans  agree musigma  agree frozen  regret hp musigma  regret hp frozen
           all                  all      8594          0.320         0.269              8.330             9.654
      scenario                   S1       832          0.428         0.429              7.115             7.543
      scenario                   S2      1433          0.400         0.311              7.494             8.969
      scenario                   S3      2060          0.282         0.239              6.063             7.120
      scenario                   S4      2594          0.241         0.198              9.462            11.114
      scenario                   S5      1675          0.365         0.299             10.684            12.146
         phase       before contact       445          0.243         0.317             17.615            15.325
         phase first 3 s of contact      2000          0.495         0.399             10.843            14.749
         phase                later      6149          0.268         0.223              6.841             7.587
red_sd_tercile              compact      2865          0.180         0.205              6.661             6.995
red_sd_tercile               middle      2864          0.397         0.308              7.478             9.271
red_sd_tercile               spread      2865          0.382         0.294             10.851            12.696
      decisive                False      8283          0.322         0.262              7.522             9.147
      decisive                 True       311          0.267         0.447             29.841            23.173
```

## The oracle's picks

```
advance/nearest     0.375
hold/nearest        0.287
advance/focus       0.225
withdraw/nearest    0.103
hold/focus          0.010
```

## Confusion: oracle pick (rows) vs musigma pick (columns)

```
musigma           advance/focus  advance/nearest  hold/focus  hold/nearest  withdraw/nearest
oracle                                                                                      
advance/focus               509              347         494           492                90
advance/nearest             847              632         870           648               226
hold/focus                   20               25           8             5                24
hold/nearest                762              546          40           284               838
withdraw/nearest            196              244          84           164               199
```

## Confusion: oracle pick (rows) vs frozen pick (columns)

```
frozen            advance/focus  advance/nearest  hold/focus  hold/nearest  withdraw/nearest
oracle                                                                                      
advance/focus               922                4         871             3               132
advance/nearest            1554               65        1321             7               276
hold/focus                   61                0          10             0                11
hold/nearest               1430               42         104           143               751
withdraw/nearest            198                9         360             7               313
```

## Read-outs (facts)

- 8,594 re-plans, replay identical to `43` (0 mismatches).
- **Agreement with the perfect model:** μ, σ 32%, frozen 27%; with three distinct choices in contact, chance is about
  33%. **Mean regret:** μ, σ 8.3 hp, frozen 9.7 hp per re-plan.
- **The perfect model picks a nearest intent in 66% of re-plans** (advance/nearest 37.5%, hold/nearest 28.7%); focus
  23.5%; withdraw 10.3%. The frozen model picks focus in about 85% of re-plans; the μ, σ model in about 51%.
- When the oracle picks hold/nearest, the μ, σ model picks withdraw/nearest in 838 of 2,470 re-plans.
- By phase: agreement is highest in the first 3 s of contact (μ, σ 50%); regret is highest before contact (15-18 hp).
- On decisive re-plans (top two more than 20 hp apart, 3.6%), the regret is 23-30 hp, and the frozen model agrees
  more often (45%) than the μ, σ model (27%).
