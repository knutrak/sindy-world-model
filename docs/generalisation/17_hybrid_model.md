# A unit-level hybrid world model: results (23 Nov 2026)

Plan: `16_hybrid_model_plan.md` (approved 23 Nov). Script: `experiments/generalisation/hybrid_model.py`; model:
`src/sindy_wm/models/hybrid.py`. Val: 300 battles per scenario (exp06, exp08), start points every 8th step, paired
with the frozen pipeline on the same start points.

```
scenario   h               quantity  hybrid  frozen  ratio  difference                95%  below 0
 marines 1.0 H1 reach error (units)  2.8476  1.7582 1.6196      1.0894 [+0.9245, +1.2542]    False
 marines 1.0        H2 margin error  0.0232  0.0208 1.1172      0.0024 [+0.0014, +0.0034]    False
 marines 2.0 H1 reach error (units)  2.6814  2.0314 1.3200      0.6500 [+0.4874, +0.8163]    False
 marines 2.0        H2 margin error  0.0436  0.0362 1.2061      0.0075 [+0.0052, +0.0097]    False
 marines 4.0 H1 reach error (units)  2.4074  2.2094 1.0896      0.1980 [-0.0084, +0.3957]    False
 marines 4.0        H2 margin error  0.0899  0.0752 1.1964      0.0148 [+0.0093, +0.0205]    False
   mixed 1.0 H1 reach error (units)  2.5305  1.6184 1.5635      0.9121 [+0.7880, +1.0336]    False
   mixed 1.0        H2 margin error  0.0321  0.0250 1.2829      0.0071 [+0.0061, +0.0081]    False
   mixed 2.0 H1 reach error (units)  2.8680  2.0236 1.4173      0.8444 [+0.6874, +1.0036]    False
   mixed 2.0        H2 margin error  0.0700  0.0422 1.6573      0.0278 [+0.0249, +0.0308]    False
   mixed 4.0 H1 reach error (units)  2.9066  2.1492 1.3524      0.7574 [+0.5906, +0.9317]    False
   mixed 4.0        H2 margin error  0.1507  0.0775 1.9452      0.0732 [+0.0648, +0.0814]    False
```

| stage | rule | result |
|---|---|---|
| **H1 geometry** | reach error ≤ 0.7 × closure at 2 s and 4 s | **fails**: 1.32 × and 1.09 × (marines), 1.42 × and 1.35 × (mixed): worse than the closure |
| H2 full model | margin error below frozen at 2 s and 4 s | fails: 1.21 × / 1.20 × (marines), 1.66 × / 1.95 × (mixed) |
| H3, H4 | run only if the earlier stages pass | **not run** |

Facts:
- Reach at the start point is reproduced exactly (same units, same rule); the error builds up within 1 s.
- In an inspected battle (exp06 val, episode 3, start step 24), the model had 11 Red units in range after 2 s against
  4 in the game, and Blue's health fell faster than recorded.
- The model leaves out collision avoidance: simulated units walk through each other and reach their targets
  together, while real units queue and arrive piecemeal (the effect the oracle's step-back exploits, `planning/19`).
  `15` found 11-28% of unit-steps slowed or deflected by crowding; this result shows that share is what makes the
  geometry right, not noise around it.

**Reading for next semester (not tested):** a unit-level world model needs crowding. A minimal version keeps units
apart (no overlap closer than the sum of their radii, as SMAClite's collision avoidance does), which is a known rule,
not a fitted one. Whether a hybrid law with such a separation step predicts reach better than the closure is the
natural first experiment.
