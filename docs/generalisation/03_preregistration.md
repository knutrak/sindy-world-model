# Generalisation test: pre-registration (21 Nov 2026)

Written and committed **before** any held-out table is built or read. The script
`experiments/generalisation/heldout_test.py` is committed in the same commit and runs **once**. Anything
learned afterwards is post-hoc.

## What is tested

The frozen pipeline (`configs/generalisation_pipeline.yaml`: withdraw rule, proportional target choice,
STLSQ 0.15 / 0.05, reach capped at 6, SG window 9) in four held-out sets:

| scenario | held-out set | model scored on it |
|---|---|---|
| marines | `exp06_intents/test4` (1000 battles, collected 20 Nov) | fitted on exp06 train |
| marauders | `exp09_marauders/test` (500, collected 19 Nov) | fitted on exp09 train |
| mixed | `exp08_mixed/test3` (1000, collected 19 Nov) | fitted on exp08 train |
| **uneven** (never seen) | `exp10_uneven/test` (1000, collected 20 Nov) | the **mixed** model, unchanged |

Two uses of the held-out data:

- **Prediction:** the train-fitted models are rolled out on the held-out battles.
- **Discovery:** the pipeline is also *refitted* on each held-out set on its own, to check what it discovers
  there (structure, physics). For uneven this is the pipeline's first contact with a new scenario.

Baselines, fitted on the same train sets: persistence, Lanchester square (team totals), and for mixed and
uneven the team-level A*-2k. Paired 95% intervals: 1000 battle resamples.

Knut's use case needs 1-2 s, so **2 s is the main horizon**. 4 s is checked as well.

## Expectations (each reported as met or not met)

1. **Structure (refit on each held-out set).** The health equations keep exactly one firepower term per
   attacker type and target type, no other health terms, and no sign failures (G1, G2). The distance equation
   keeps at most 6 terms.
2. **Physics (refit on each held-out set).**
   - Every firepower coefficient is 0.75-1.0 × its game rate.
   - Every walking-speed term (advance, hold in contact, hold far, withdraw) is within 20% of the game's 6.3 /
     3.15. The plan's draft said 15%; on val the range is 0.84-1.09, so 20% is set here, before any held-out read.
3. **Prediction against the baselines (train-fitted models).** In each scenario, the health error is below
   persistence and Lanchester at 2 s and at 4 s: paired 95% interval of (model − baseline) entirely below 0.
4. **Per-type against team-level (mixed, uneven).** The per-type model's health error at 2 s is below the
   team-level A*-2k's (paired 95% interval entirely below 0).
5. **Reliable horizon (G6, train-fitted models).** At least 2 s in marines, marauders and mixed. For uneven it
   is reported, not expected.
6. **Held-out ≈ val (G7-like).** Health error at 2 s on held-out ≤ 1.2 × the val value, for marines (val
   0.0279), marauders (0.0203) and mixed (0.0346).

Reported, not judged: margin R² at 2 / 4 s, battle nRMSE, winner accuracy, alive and distance errors, the
uneven scenario's values against the mixed val values.

## Rules

- No refit, retune or second run after the held-out tables are built. Problems found afterwards are reported
  as post-hoc.
- If the script crashes before producing results, it may be fixed and run again only if no held-out result has
  been printed. Any such fix is logged.
