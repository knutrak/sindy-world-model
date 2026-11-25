# Every model fitted in the project (25 Nov 2026)

Where each came from is in the linked note. Health error = mean absolute error per side at 4 s; scales differ between
blocks (marines: 1 = ten marines; marauders: ten marauders), so compare within a block only. k in full units per second
per unit in reach (game maximum: marine 0.213, marauder 0.135, stalker 0.079).

## Marines only

| model | what | terms | health 4 s | status |
|---|---|---|---|---|
| polynomial SINDy on health (exp03) | −0.36 B + 0.20 R + ... | 3-12 | other data | superseded: predicts healing |
| gated SINDy (exp03, exp05) | −R·p(B, R) | 6-10 | other data | superseded: terms change sign between data sets |
| firepower law (exp05) | losses = 0.20 × engaged enemy units | 1 | — | found by choosing the variable, not by SINDy |
| generic SINDYc, degree 2 | polynomial with intents | 138 | 0.053 | reference: as accurate, not readable |
| switch library (`sindyc_04`) | SINDy picks the switches | 4 | 0.058 | exploratory |
| A (`model_derived`) | reach + derived closure; k 0.177 / 0.186, p 0.52 / 0.51 | 11 | 0.053 cv | fails gate G3b on 17 Nov |
| W, WG | A in weak form | 12 | about A | fails G3b / no gain |
| B | A's terms fitted on rollouts; k 0.156 / 0.165 | 11 | 0.050 cv | only valid interpretable model on 17 Nov |
| S | B + share in reach as a state | 22 | 0.048 cv | predictive extension; 2 wrong signs |
| A* | one k (0.182) + reach alive law | 10 | 0.051 cv | passed test2 expectations |
| A*-2k | k 0.177 / 0.186, p 0.66 / 0.54 | 11 | 0.051 cv | final marine model 19 Nov; no own held-out run |
| **pipeline, marines** | same k; withdraw by rule; p 0.66 / 0.56 | 10 | **0.051 test4** | **frozen, tested** |
| trees / MLP | black boxes | 5-19 k weights | 0.032 / 0.033 | references |

Also screened on cv (18 Nov): other optimisers (SR3, SSR, FROLS), E-SINDy, phase hybrid, hazard death law, size-scaled
library, symmetric k. None replaced A's family.

## Marauders only (`mixed/PROGRESS.md`, exp09)

| model | k Blue-loss / Red-loss | p | status |
|---|---|---|---|
| A*-2k, reach 7.125 | 0.077 / 0.086 | 0.63 / 0.53 | wrong reach: 57-64% of the game |
| marine A*-2k scaled, reach 6 | 0.115 / 0.121 | 0.66 / 0.54 | transfers (margin R² 2 s 0.36) |
| A*-2k refit, reach 6 | 0.125 / 0.122 | 0.61 / 0.54 | |
| **pipeline, marauders** | 0.125 / 0.126 | 0.61 / 0.54 | **frozen, tested** (health 4 s 0.037) |

## Stalkers only (exp07)

| model | what | status |
|---|---|---|
| marine models scaled (A, B, S) | health 4 s 0.040 / 0.031 / 0.028 (test2) | transfer without refit |
| A-type refits | 0.024-0.026 val | exploratory |
| **pipeline, stalkers** (25 Nov) | k 0.068 / 0.069 (0.85 / 0.87 × game), p 0.66 / 0.55 | val only; closing speed wrong (`triple/04`) |

## Marines + marauders (exp08)

| model | what | status |
|---|---|---|
| team A*-2k | one k per side 0.218 / 0.220 (reach 6) | baseline |
| TypedA, closure / proportional, uncapped | 4 k per side; marauder terms 0.60-0.71 × game | superseded |
| TypedA, reach 6, ± withdraw rule | 0.82-0.94 × game | superseded by the pipeline |
| **pipeline, mixed** | 8 firepower terms (0.060-0.189), 19 terms | **frozen, tested** (test3 and uneven exp10) |
| pipeline + kite (exp11) | no kite term in health; kite gate 5.47 | K3 failed |

Firepower, pipeline mixed (Blue-loss / Red-loss): marine → marine 0.178 / 0.189, marauder → marine 0.165 / 0.172,
marine → marauder 0.060 / 0.060, marauder → marauder 0.113 / 0.117.

## Marines + marauders + stalkers (exp12, 25 Nov)

| model | what | status |
|---|---|---|
| **pipeline, triple** | 18 firepower terms, 0.78-1.01 × game; p 0.35-0.48; 5 distance terms | 7 of 7 expectations on test; structure fragile (`triple/04`) |

## Variants of the pipeline (marines and mixed; `planning/`, `generalisation/`)

| variant | state or law added | result |
|---|---|---|
| focus cost, focus / shape terms | short-term focus cost; intent terms in d' and shape | no gain in play |
| spread | evenness in the alive law | tiny val gain, same play |
| compartments, bolted on / discovered | healthy / wounded / dead | no better rollouts or play |
| reach as a state | generic library, weak form | 54-232 groups: not sparse |
| μ, σ margins | mean and spread of reach margins | 26-51 groups; best SINDy planner, below fixed kite |
| typed MLP | black box | best offline, worst planner |
| per-unit sparse law | velocity regression | R² 0.13-0.57: fails |
| unit-level hybrid (± separation, steering) | walk / stop switch + frozen damage | matches the reach closure at best |
