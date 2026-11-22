# Black-box world model in the same planner: results

Plan: `36_black_box_planner_plan.md` (approved 22 Nov). Script: `experiments/planning/black_box.py`.

## Gate B0 (counterfactual ranking): passes

```
scenario   h  regret frozen  regret mlp  regret mu-sigma               mlp − frozen  below frozen             mlp − mu-sigma  regret always kite_focus
 marines 1.0         0.0170      0.0052           0.0109 -0.0118 [-0.0138, -0.0099]          True -0.0058 [-0.0074, -0.0042]                       NaN
 marines 2.0         0.0225      0.0097           0.0153 -0.0128 [-0.0159, -0.0100]          True -0.0056 [-0.0082, -0.0031]                       NaN
 marines 4.0         0.0308      0.0191           0.0203 -0.0117 [-0.0160, -0.0069]          True -0.0012 [-0.0046, +0.0020]                       NaN
   mixed 1.0         0.0281      0.0158           0.0217 -0.0123 [-0.0148, -0.0097]          True -0.0059 [-0.0084, -0.0035]                    0.0269
   mixed 2.0         0.0516      0.0292           0.0341 -0.0224 [-0.0268, -0.0181]          True -0.0049 [-0.0091, -0.0007]                    0.0461
   mixed 4.0         0.0887      0.0602           0.0669 -0.0284 [-0.0360, -0.0205]          True -0.0066 [-0.0147, +0.0015]                    0.0672
```

## Val errors

```
scenario    model   h  health error  margin error  margin R²
 marines   frozen 1.0        0.0327        0.0209     0.4308
 marines   frozen 2.0        0.0554        0.0358     0.4012
 marines   frozen 4.0        0.1046        0.0745     0.0798
 marines      mlp 1.0        0.0265        0.0175     0.6038
 marines      mlp 2.0        0.0372        0.0259     0.6960
 marines      mlp 4.0        0.0679        0.0489     0.6064
 marines mu-sigma 1.0        0.0337        0.0209     0.4126
 marines mu-sigma 2.0        0.0577        0.0361     0.3743
 marines mu-sigma 4.0        0.1083        0.0758     0.0192
   mixed   frozen 1.0        0.0418        0.0268     0.3973
   mixed   frozen 2.0        0.0719        0.0448     0.3869
   mixed   frozen 4.0        0.1330        0.0827     0.1887
   mixed      mlp 1.0        0.0366        0.0248     0.5027
   mixed      mlp 2.0        0.0563        0.0378     0.5865
   mixed      mlp 4.0        0.0979        0.0659     0.5432
   mixed mu-sigma 1.0        0.0424        0.0271     0.3755
   mixed mu-sigma 2.0        0.0727        0.0475     0.3121
   mixed mu-sigma 4.0        0.1412        0.1026    -0.2432
```

## Closed loop: wins per policy (34's seeds)

```
                        S1     S2     S3     S4     S5
policy                                                
fixed:hold/focus     0.685    NaN  0.610  0.070    NaN
fixed:kite/focus     0.972  0.855  0.990  0.242  0.602
fixed:kite/nearest   0.968  0.895  0.992  0.290  0.615
mlp:h3               0.365  0.348  0.570  0.098  0.258
model:h3             0.652  0.628  0.735  0.070  0.560
musigma:h3           0.675  0.800  0.745  0.200  0.575
fixed:advance/focus    NaN  0.635    NaN    NaN  0.555
```

## Paired comparisons (MLP planner − reference)

```
scenario rule      against  difference              95% read-out   n
      S1   D3 kite/nearest      -0.602 [-0.650, -0.552]    loses 400
      S1   D2   musigma:h3      -0.310 [-0.362, -0.260]    loses 400
      S1   D3   hold/focus      -0.320 [-0.370, -0.268]    loses 400
      S2   D1 kite/nearest      -0.548 [-0.600, -0.497]    loses 400
      S2   D2   musigma:h3      -0.452 [-0.510, -0.395]    loses 400
      S3   D3   kite/focus      -0.420 [-0.470, -0.370]    loses 400
      S3   D2   musigma:h3      -0.175 [-0.237, -0.110]    loses 400
      S3   D3   hold/focus      -0.040 [-0.107, +0.025]     ties 400
      S4   D1 kite/nearest      -0.192 [-0.235, -0.150]    loses 400
      S4   D2   musigma:h3      -0.102 [-0.142, -0.062]    loses 400
      S5   D1 kite/nearest      -0.358 [-0.405, -0.310]    loses 400
      S5   D2   musigma:h3      -0.318 [-0.365, -0.270]    loses 400
```

**Interpretation fixed in the plan: the setup limits planning (D1 loses in every main scenario).**

## Planner diagnostics

```
scenario  pick_advance/nearest  pick_advance/focus  pick_hold/nearest  pick_hold/focus  pick_withdraw/nearest  near_tie_share  switches  wall_s  pick_kite/nearest  pick_kite/focus
      S1                 0.222               0.140              0.269            0.264                  0.104           0.510     4.562   0.622                NaN              NaN
      S2                 0.062               0.015              0.134            0.220                  0.209           0.447     6.040   1.606              0.176            0.182
      S3                 0.037               0.030              0.078            0.196                  0.178           0.379     6.402   1.415              0.096            0.386
      S4                 0.151               0.022              0.164            0.077                  0.069           0.270     5.222   0.856              0.314            0.204
      S5                 0.122               0.021              0.199            0.125                  0.174           0.297     5.688   1.432              0.164            0.194
```

## Read-outs against the plan (`36`)

| | rule | result |
|---|---|---|
| B0 | MLP ranks intents better than frozen (2 s, 4 s) | **passes**: regret 35-57% lower than frozen and also lower than the μ, σ model; val margin R² at 4 s 0.61 (marines) / 0.54 (mixed), against 0.08 / 0.19 frozen |
| **D1** | MLP planner against best fixed kite, S2, S4, S5 | **loses in all three**: −54.8, −19.2, −35.8 win points |
| D2 | MLP planner against the μ, σ planner | **loses in all five scenarios** (−10.2 to −45.2) |
| D3 | flagged | S1 loses to kite (−60.2) and hold/focus (−32.0); S3 loses to kite, ties hold/focus |

**Interpretation fixed in the plan: the setup limits planning.**

Facts beside the rules:
- The black-box planner is also below the frozen SINDy planner in S1, S2, S3 and S5 (e.g. S2 34.8% vs 62.8%); S4
  9.8% vs 7.0%.
- It picks withdraw in 7-21% of re-plans and hold/nearest in 8-27%, more than the SINDy planners, and switches 5-6
  times per battle.
- Better open-loop accuracy and ranking (B0, recorded counterfactual sets) did not carry into closed loop. Not
  checked: whether the planner's own states and intent sequences lie outside what the MLP was trained on, so that
  the planner picks intents whose predicted gains are model errors (a known failure of planning with learned
  models). Start points at step 0 were also not in the MLP's training starts (history 8 steps).
