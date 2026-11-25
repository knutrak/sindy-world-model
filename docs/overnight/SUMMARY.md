# Overnight experiment: summary (25 Nov 2026)

Plan `00_plan.md`; search `01_search.md`; blind test `02_blind_test.md`; planning `03_planning.md`. Finalists and scripts were
committed (4083f89) before any new battle. Finished at about 20:20, far ahead of the estimate.

| stage | result |
|---|---|
| search | 51 variants, train only, left-out unit types. H5 -3.0%, I7 -2.1% transfer error against B1 |
| blind test (read once) | H5 and I7 meet T1-T5; B1 (marines only) 0.97 and 1.04 x own on marauders, stalkers (limit 1.10) |
| planning (400 battles per scenario and policy) | L1 and L2 **not met** by B1, H5 or I7 |

- T1: H5 3 of 4 (worse on stalkers), I7 4 of 4. T2: H5 2 of 3 (worse on n3), I7 3 of 3.
- Planning: against the frozen planner B1 and H5 beat it in S3 (+8.0, +5.2 points) and tie elsewhere; I7 ties all. Against the
  best fixed intent all win only S3; B1 and I7 lose S1. B1 has the most wins in S3 (0.83, matched perfect model 0.93).
- Knut's expectation, "more accurate planning results and slightly better prediction results": prediction met; planning not.

Not done: agreement with the perfect model and optimism of chosen plans (planned as reported items).
Caveats: the live closing speed (unit velocities) correlates 0.75 with the recorded backward difference the models were scored on;
"H5 LO" has 23 terms against 18 for "H5 all"; unit properties are inputs, so firepower is not a check against the game;
three planners were compared in five scenarios with no correction for that; a model's own fits (B0) use their own validation.
