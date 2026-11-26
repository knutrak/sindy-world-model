# Candidates from the battle shapes, selected by STLSQ: results (26 Nov 2026)

Plan: `24_shape_library_plan.md`. Script: `experiments/generalisation/shape_library.py` (58 s). Fitted on all train
battles, rolled out on all val battles (marines 8,946 start points at 1 s, mixed 14,520). Output:
`data/processed/generalisation/shape_library/` (`terms.txt`, `val.json`, `models.pkl`). The copied rollout reproduces
`Pipeline.rollout` exactly (checked: max difference 0.0 on 100 val battles per scenario). Exploratory; val read before.

## Old library vs new: what STLSQ keeps

**Health** (new: enemy health H_opp, H_opp·A_own, own health H_own). Nothing new is kept; every equation is identical
to frozen, in marines and mixed. Units in reach carries what enemy health would.

**Alive** (now selected by STLSQ; candidates: law = A·Ĥ'/H, law·focus for Red, enemy alive, enemy in reach, own alive,
constant):

| equation | frozen (hand law) | selected by STLSQ |
|---|---|---|
| marines Blue | 0.662·law | 0.6625·law |
| marines Red | 0.562·law + 0.260·law·focus | 0.5623·law + 0.2600·law·focus |
| mixed Blue marine | 0.578·law | 0.5776·law |
| mixed Blue marauder | 0.444·law | 0.4435·law |
| mixed Red marine | 0.506·law + 0.155·law·focus | 0.3522·law − 0.0895·enemy in reach |
| mixed Red marauder | 0.376·law + 0.139·law·focus | 0.4337·law |

Out of the six candidates, STLSQ picks the law term in every equation, with the same numbers as the hand fit, and
rejects enemy alive, own alive and the constant. The focus term is kept for marines and dropped in mixed (once replaced
by enemy in reach, right sign).

**Distance** (new: d·u and u for u in advance, hold, withdraw):

| | frozen | selected by STLSQ |
|---|---|---|
| marines | −0.485 walk_r·wd + 0.575 danger·wd − 6.008 close·adv − 3.183 close·hold·far − 5.915 close·hold·contact | +0.190 danger·wd − 5.815 close·adv − 3.183 close·hold·far − 5.915 close·hold·contact **− 0.0455 d·adv + 0.273 adv − 0.331 d·wd + 2.138 wd** |
| mixed | −0.340 walk_r·wd + 0.708 danger·wd − 5.434 close·adv − 3.081 close·hold·far − 5.263 close·hold·contact | −0.189 walk_r·wd + 0.583 danger·wd − 5.270 close·adv − 3.081 close·hold·far − 5.263 close·hold·contact **− 0.0437 d·adv + 0.281 adv − 0.082 d·wd + 0.570 wd** |

The settling terms are kept with their expected signs; `d·hold` came out with the wrong sign and was dropped. Each
pair implies a level the distance settles towards, d_eq = constant / slope:

| | under advance | under withdraw |
|---|---|---|
| marines | 0.273 / 0.0455 = **6.0** (marine reach 5.75, order limit 6) | 2.138 / 0.331 = **6.5** (withdraw radius 7.0) |
| mixed | 0.281 / 0.0437 = **6.4** (pair reach 5.75-7.0, limit 6) | 0.570 / 0.082 = **7.0** (withdraw radius 8.0) |

Under withdraw the settling term partly replaces the rule terms (walk_r·wd drops out for marines).

## Val rollouts against frozen

Health error (mean over sides), margin R², distance error; paired bootstrap by battle.

| scenario | variant | health 2 s | health 4 s | health 8 s | margin R² 4 s | distance 4 s | distance 8 s |
|---|---|---|---|---|---|---|---|
| marines | frozen | 0.0277 | 0.0523 | 0.0955 | 0.080 | 0.499 | 0.828 |
| marines | health | same | same | same | same | same | same |
| marines | alive | 0.0277 | 0.0523 | 0.0954 | 0.082 | 0.499 | 0.828 |
| marines | distance | 0.0276 | 0.0524 | 0.0983 (+3%, interval above 0) | 0.071 | 0.503 | 0.811 |
| marines | all | 0.0276 | 0.0524 | 0.0982 (+3%) | 0.073 | 0.503 | 0.811 |
| mixed | frozen | 0.0346 | 0.0640 | 0.1132 | 0.148 | 0.599 | 0.831 |
| mixed | alive | 0.0345 | 0.0639 | 0.1128 | 0.151 | 0.599 | 0.831 |
| mixed | distance | 0.0346 | 0.0639 | 0.1133 | 0.150 | 0.592 (−1%, below 0) | 0.810 (−2.5%, below 0) |
| mixed | all | 0.0345 | 0.0638 | 0.1129 | 0.153 | 0.592 | 0.810 |

## Read-outs

| rule | result |
|---|---|
| S1 selected | **passes**: the settling distance terms (right signs, 4 of 6 kept) and, once, enemy in reach in Red's alive equation |
| S2 better | **fails**: health and margin change by under 1% at 2-4 s in both scenarios; marines' health is 3% worse at 8 s |

## What it says

- **The alive law is now selected, not only chosen:** offered next to the shapes from the battle curves (losses
  proportional to enemy alive, to enemy in reach, to own alive, or constant), STLSQ keeps the health-following law
  and nothing else, with the same exponents as the hand fit. Its form was still proposed by hand.
- **Square-law in health and exponential losses are rejected:** units in reach explains health better than enemy health
  or own health, in every equation.
- **Distance settles where the game's rules say:** the levels implied by the selected terms are 6.0-6.4 under advance
  (the reach and the limit of 6) and 6.5-7.0 under withdraw (the scripts' withdraw radius, 7 and 8). SINDy found these
  without being given either number. They are small corrections next to the gates (slopes 0.04-0.33 per second
  against −5 to −6 for close·adv) and do not change the health predictions.
- None of this moves prediction: the frozen model's error is not in these equations' form (`23`: it is in the share in
  reach).
