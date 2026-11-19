# The 17 Nov gates applied to "A*, separate k" (POST-HOC, 19 Nov)

Same procedure and sets as `04_gates_post_hoc.md` (val, cf_*, exp07 val). G7 not evaluated (no
test2 run). Script: `scripts/gates_separate_k.py`; every number: `05_gates_separate_k.json`.

```
G1                                                             True
G2                                                             True
G3a                                                            True
G3b                                                            True
G4                                                             True
G5                                                             True
G7                                                             None
G8                                                             True
G6 horizon val / cf                                           2 / 3
verdict                valid interpretable model (G7 not evaluated)
transfers                                                      True
```

- terms: {'health_blue': {'Rr': -0.1770129786509026}, 'health_red': {'Rb': -0.18613803068287188, 'Rb*wd': 0.16327721535893075}, 'distance': {'close*adv': -6.007670902867294, 'close*wd': -3.1575425101087875, 'flee*wd': 0.2961645741299556, 'close*hold*far': -3.1830209969861674, 'close*hold*contact': -5.914738967135064}, 'alive_blue': {'1': 0.6624848921668055}, 'alive_red': {'1': 0.5394095781023318, 'focus': 0.23843304006641233}}
- G2 sign failures: none
- G3a terms failing: none
- G3b: {'same term set': True, 'term sets that differ': {}, 'health coefficients': {'health_blue: Rr': {'values': [-0.1863, -0.177, -0.1742, -0.1824, -0.184], 'max deviation from median': 0.04510484180919867}, 'health_red: Rb': {'values': [-0.1934, -0.1861, -0.1832, -0.1923, -0.1944], 'max deviation from median': 0.04740564827811566}, 'health_red: Rb*wd': {'values': [0.1783, 0.1633, 0.1541, 0.1733, 0.1698], 'max deviation from median': 0.09281789574895509}}, 'alive exponents': {'SG 5': {'alive_blue': {'1': 0.6443}, 'alive_red': {'1': 0.5155, 'focus': 0.2364}}, 'SG 9': {'alive_blue': {'1': 0.6625}, 'alive_red': {'1': 0.5394, 'focus': 0.2384}}, 'SG 17': {'alive_blue': {'1': 0.6801}, 'alive_red': {'1': 0.5668, 'focus': 0.2233}}, 'weak 1 s': {'alive_blue': {'1': 0.7}, 'alive_red': {'1': 0.5677, 'focus': 0.2653}}, 'weak 2 s': {'alive_blue': {'1': 0.7182}, 'alive_red': {'1': 0.5826, 'focus': 0.2405}}}}
- G5: {'persistence': [-0.18928959946887486, -0.1704292347142181], 'Lanchester': [-0.05261739834189708, -0.04594342854662954], 'no control': [-0.027701137714361973, -0.02323440858902378]}
- G6: {'val': 2, 'cf': 3}
- G8: {'persistence': [-0.04547040349002221, -0.03818366999641011], 'no control': [-0.0195514526544756, -0.015857679041297375]}
- vs A* on val: {'health a': [0.050946187690411385, 0.05534725677702722], 'health b': [0.051156996657227526, 0.055580022048705435], 'health b-a': [2.8463919752853497e-05, 0.0004238660068933497], 'margin a': [-0.0066733349520495516, 0.18710598628390288], 'margin b': [-0.009993133988049184, 0.1733309561110936], 'margin b-a': [-0.021331260336901355, 0.004437429563917967]}
