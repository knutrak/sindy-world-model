# Hybrid movement plus separation: results (23 Nov 2026)

Plan: `18_separation_plan.md` (approved 23 Nov). Model: `src/sindy_wm/models/hybrid.py` with `separation = True`;
script: `HYBRID_SEPARATION=1 experiments/generalisation/hybrid_model.py h12`. Same val start points as `17`.

```
scenario   h               quantity  hybrid  frozen  ratio  difference                95%  below 0
 marines 1.0 H1 reach error (units)  2.7223  1.7582 1.5483      0.9640 [+0.8106, +1.1168]    False
 marines 1.0        H2 margin error  0.0215  0.0208 1.0326      0.0007 [-0.0001, +0.0014]    False
 marines 2.0 H1 reach error (units)  2.5795  2.0314 1.2698      0.5481 [+0.3854, +0.7140]    False
 marines 2.0        H2 margin error  0.0384  0.0362 1.0624      0.0023 [+0.0002, +0.0042]    False
 marines 4.0 H1 reach error (units)  2.3631  2.2094 1.0696      0.1537 [-0.0579, +0.3539]    False
 marines 4.0        H2 margin error  0.0778  0.0752 1.0351      0.0026 [-0.0024, +0.0078]    False
   mixed 1.0 H1 reach error (units)  2.5823  1.6184 1.5955      0.9638 [+0.8193, +1.1197]    False
   mixed 1.0        H2 margin error  0.0264  0.0250 1.0553      0.0014 [+0.0007, +0.0021]    False
   mixed 2.0 H1 reach error (units)  2.8833  2.0236 1.4249      0.8597 [+0.6622, +1.0694]    False
   mixed 2.0        H2 margin error  0.0518  0.0422 1.2272      0.0096 [+0.0076, +0.0117]    False
   mixed 4.0 H1 reach error (units)  2.8720  2.1492 1.3363      0.7228 [+0.5160, +0.9436]    False
   mixed 4.0        H2 margin error  0.1107  0.0775 1.4285      0.0332 [+0.0273, +0.0388]    False
```

| stage | rule | result |
|---|---|---|
| **H1 geometry** | reach error ≤ 0.7 × closure at 2 s and 4 s | **fails**: marines 1.27 × and 1.07 ×, mixed 1.42 × and 1.34 × (without separation, `17`: 1.32 / 1.09 and 1.42 / 1.35) |
| H2 full model | margin error below frozen at 2 s and 4 s | fails, but much closer: marines 1.06 × and 1.04 × (`17`: 1.21 / 1.20); mixed 1.23 × and 1.43 × (`17`: 1.66 / 1.95) |
| H3, H4 | only if the earlier stages pass | not run |

Facts:
- Separation removes the pile-up's effect on damage: in marines the model is now nearly level with the frozen pipeline
  on the health margin (interval at 4 s spanning 0).
- Reach barely improves and is already 1.5 × the closure's error at 1 s, when the closure is still anchored to the
  measured start (its offset decays over 1 s). The hybrid model also starts from the measured reach, so its error
  builds up within the first second, from per-unit walk/stop and target choices.
- Pushing units apart is not what SMAClite does (it steers them around each other); the remaining difference between
  the two is not measured here.

**Reading (not tested):** the unit-level model now gets damage about right in marines but not who is in range at each
moment. Per-unit switch errors (about 5% per unit-step) and simplified targeting compound over 8 steps and 20 units.
Candidates for next semester: learn the switch from sequences (not single steps), copy SMAClite's steering (RVO) as
a known rule, or keep the closure for reach and use unit-level movement only to update it.
