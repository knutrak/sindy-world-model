# Candidates from the battle shapes, selected by STLSQ: plan (26 Nov 2026)

Exploratory, asked by Knut after `docs/checks/curve_shapes.md`. Written before any fit. Train fit, val rollouts
(marines exp06, mixed exp08), the frozen pipeline (`generalisation_frozen.pkl`) as reference. Script:
`experiments/generalisation/shape_library.py`.

## New candidates (old candidates kept, frozen thresholds)

| finding (curve_shapes) | candidate | equation | expected sign |
|---|---|---|---|
| square law in health (R² 0.96-0.97 with 2 numbers) | H_opp (enemy health, split by the fire share) | health | − |
| fire per enemy unit falls with army size | H_opp·A_own | health | + |
| exponential loss | H_own | health | − or dropped |
| loss in units scales with enemy size | A_opp, R_opp (enemy alive / in reach, split by fire share) | alive | − |
| alive follows health (frozen law) | law = A·Ĥ'/H; law·focus (Red) | alive | + |
| linear decline | constant | alive | − or dropped |
| distance settles to d_eq ≈ 5.5, τ ≈ 1.3 s | d·u, u for u in adv, hold, wd | distance | d·adv, d·hold −; adv, hold + |

The alive equation is now selected by STLSQ (threshold 0.15) instead of the hand law; it is clipped so that alive
never rises and is 0 when health is 0. Ĥ' in the law term is the model's health rate (measured states when fitting).
Sign rule as in `22`: a kept term with the wrong sign is dropped and the rest refitted.

## Variants and read-outs (val, horizons 1, 2, 4, 8 s)

Frozen; new health; new alive; new distance; all three. Report the kept terms (old library vs new), and per variant
health error (mean over sides), margin R², distance error, alive error, with paired bootstrap against frozen.

- **S1 selected:** at least one new candidate kept with its expected sign.
- **S2 better:** health error and margin error below frozen at 2 and 4 s (95% interval below 0) in both scenarios.

Expectation (Claude): health keeps units in reach (H_opp adds little, since units in reach carry the same and more);
alive keeps the law term, maybe with enemy in reach; distance may keep the settling terms, since d_eq ≈ reach is a
clean physical pattern. S2 most likely, if anywhere, for distance.
