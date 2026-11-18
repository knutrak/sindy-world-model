# The 17 Nov gates applied to A* and A (POST-HOC, 18 Nov)

**Not a pre-registered test.** `final` (17 Nov) and `test2` (18 Nov) are spent; this applies the
gates' definitions to `val`, `cf_*` and exp07 `val`, as `evaluate_final.py --dry-run` does. A* was
designed after looking at cv and val results. G7 uses the test2 / val ratio from the Phase 3 run.
Script: `scripts/gates_post_hoc.py`; every number: `04_gates_post_hoc.json`.

```
                        A                         A*
G1                   True                       True
G2                   True                       True
G3a                  True                       True
G3b                 False                       True
G4                   True                       True
G5                   True                       True
G7                   True                       True
G8                   True                       True
G6 horizon val          2                          2
G6 horizon cf           3                          3
verdict         not valid  valid interpretable model
transfers           False                       True
```

## A

- G1 term counts: {'health_blue': 1, 'health_red': 2, 'distance': 5, 'alive_blue': 1, 'alive_red': 2}
- G2 sign failures: none
- G3a terms failing: none
- G3b: {'same term set': False, 'term sets that differ': {'SG 17': {'health_blue': ['Rr'], 'health_red': ['Rb', 'Rb*wd'], 'distance': ['close*adv', 'close*hold*contact', 'close*hold*far', 'close*wd', 'flee*wd'], 'alive_blue': ['1'], 'alive_red': ['1', 'focus', 'wd']}, 'weak 1 s': {'health_blue': ['Rr'], 'health_red': ['Rb', 'Rb*wd'], 'distance': ['close*adv', 'close*hold*contact', 'close*hold*far', 'close*wd', 'flee*wd'], 'alive_blue': ['1', 'focus'], 'alive_red': ['1', 'focus']}, 'weak 2 s': {'health_blue': ['Rr'], 'health_red': ['Rb', 'Rb*wd'], 'distance': ['close*adv', 'close*hold*contact', 'close*hold*far', 'close*wd', 'flee*wd'], 'alive_blue': ['1', 'focus'], 'alive_red': ['1', 'focus']}}, 'health coefficients': {'health_blue: Rr': {'values': [-0.1863, -0.177, -0.1742, -0.1824, -0.184], 'max deviation from median': 0.04510484180919867}, 'health_red: Rb': {'values': [-0.1934, -0.1861, -0.1832, -0.1923, -0.1944], 'max deviation from median': 0.04740564827811566}, 'health_red: Rb*wd': {'values': [0.1783, 0.1633, 0.1541, 0.1733, 0.1698], 'max deviation from median': 0.09281789574895509}}}
- G4: {'val': {'finite': True, 'non-negative': True, 'alive never above start': True}, 'cf': {'finite': True, 'non-negative': True, 'alive never above start': True}}
- G5 (paired 95%, model − baseline, health 4 s): {'persistence': [-0.18728728744550127, -0.1685183303095365], 'Lanchester': [-0.05065193333033319, -0.044040128474783946], 'no control': [-0.02573994351233318, -0.021210229402019647]}
- G6 horizon: {'val': 2, 'cf': 3}
- G7: {'health 4 s test2': 0.05388305961025505, 'val': 0.055107507387441, 'ratio': 0.9777807446710064}
- G8: {'persistence': [-0.04417350641800763, -0.036938484031133346], 'no control': [-0.01822328609686095, -0.014566960167721042]}

## A*

- G1 term counts: {'health_blue': 1, 'health_red': 2, 'distance': 5, 'alive_blue': 1, 'alive_red': 2}
- G2 sign failures: none
- G3a terms failing: none
- G3b: {'same term set': True, 'term sets that differ': {}, 'health coefficients': {'health_blue: Rr': {'values': [-0.1901, -0.1819, -0.179, -0.1877, -0.1895], 'max deviation from median': 0.04620763018649259}, 'health_red: Rb': {'values': [-0.1901, -0.1819, -0.179, -0.1877, -0.1895], 'max deviation from median': 0.04620763018649259}, 'health_red: Rb*wd': {'values': [0.175, 0.159, 0.1499, 0.1674, 0.1618], 'max deviation from median': 0.08182686346537543}}, 'alive exponents': {'SG 5': {'alive_blue': {'1': 0.6313}, 'alive_red': {'1': 0.5245, 'focus': 0.2405}}, 'SG 9': {'alive_blue': {'1': 0.6447}, 'alive_red': {'1': 0.552, 'focus': 0.244}}, 'SG 17': {'alive_blue': {'1': 0.6619}, 'alive_red': {'1': 0.58, 'focus': 0.2285}}, 'weak 1 s': {'alive_blue': {'1': 0.6804}, 'alive_red': {'1': 0.5816, 'focus': 0.2718}}, 'weak 2 s': {'alive_blue': {'1': 0.6975}, 'alive_red': {'1': 0.5979, 'focus': 0.2468}}}}
- G4: {'val': {'finite': True, 'non-negative': True, 'alive never above start': True}, 'cf': {'finite': True, 'non-negative': True, 'alive never above start': True}}
- G5 (paired 95%, model − baseline, health 4 s): {'persistence': [-0.18958045270587903, -0.17055941686935536], 'Lanchester': [-0.052885153378800516, -0.04615887469187297], 'no control': [-0.027886207586762614, -0.023512304548585463]}
- G6 horizon: {'val': 2, 'cf': 3}
- G7: {'health 4 s test2': 0.05191299037361761, 'val': 0.0529485702935199, 'ratio': 0.9804417774802687}
- G8: {'persistence': [-0.04491146646898322, -0.03751091276201724], 'no control': [-0.018929832496525592, -0.015261913806601208]}
- margin R² at 4 s, A* − A (paired 95%): [0.0021496116260317556, 0.04119636974837077]
