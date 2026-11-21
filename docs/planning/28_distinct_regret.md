# Counterfactual picks over distinct choices

Script: `experiments/planning/distinct_regret.py` (see its docstring). Counterfactual sets of S1 and S2.

```
                         states  in contact  picks differ (five candidates)  picks differ (distinct choices)  regret, frozen  regret, bolted-on  regret, discovered
scenario h   model                                                                                                                                                 
S1       1.0 bolted-on      522      0.8985                          0.5632                           0.0134          0.0170             0.0169                 NaN
             discovered     522      0.8985                          0.6877                           0.1398          0.0170                NaN              0.0154
         2.0 bolted-on      522      0.8985                          0.6054                           0.0115          0.0225             0.0224                 NaN
             discovered     522      0.8985                          0.6149                           0.0345          0.0225                NaN              0.0223
         4.0 bolted-on      522      0.8985                          0.6418                           0.0345          0.0308             0.0309                 NaN
             discovered     522      0.8985                          0.6284                           0.0785          0.0308                NaN              0.0327
S2       1.0 bolted-on      598      0.9181                          0.4749                           0.0468          0.0188             0.0184                 NaN
             discovered     598      0.9181                          0.4783                           0.1522          0.0188                NaN              0.0173
         2.0 bolted-on      598      0.9181                          0.4900                           0.0201          0.0297             0.0299                 NaN
             discovered     598      0.9181                          0.4147                           0.0401          0.0297                NaN              0.0298
         4.0 bolted-on      598      0.9181                          0.5535                           0.0468          0.0444             0.0445                 NaN
             discovered     598      0.9181                          0.4348                           0.0602          0.0444                NaN              0.0438
```

