# One equation-discovery pipeline across unit types: report (21 Nov 2026)

Branch `mixed-marauders`. Supporting files in this folder:

- `00_plan.md`: the plan.
- `PROGRESS.md`: the running log, including the decision rules.
- `01_consistency.md`: WP2.
- `02_val_table.md`: WP4 and WP5.
- `03_preregistration.md`, including its amendment.
- `04_test_results.md`: the held-out test, run once.
- `05_withdraw_movement.md`: the gates vs rule comparison.
- `06_frozen_equations.md`: the frozen models.

The work that led here (marauders, mixed armies, SMAClite's range limit) is in `docs/mixed/`.

## 1. The claim, and the short answer

> The same equation-discovery pipeline, given only a scenario's data and its units' game statistics,
> produces equations that select the same kind of terms, have coefficients that match the game's rules, and
> predict comparably well, for marines, for marauders, and for mixed marine + marauder armies.

**Short answer (hedged): supported.** In a pre-registered test, run once on four held-out sets (marines,
marauders, mixed armies, and an uneven-mix scenario the pipeline had never seen), 19 of 20 expectations were met.

- **Structure:** identical everywhere: one firepower term per attacker type and target type, five distance
  terms, no wrong signs.
- **Physics:** firepower at 0.77-1.02 of the game's rates, walking speeds at 0.81-1.04.
- **Prediction:** every model beats persistence and Lanchester at 2 and 4 s; the per-type model beats a
  team-level model in mixed armies; held-out errors match val.
- **The one miss:** a firepower term at 1.02 × the game rate in the uneven scenario (pre-registered bound 1.0).
- **What it is not:** a long-range predictor. The reliable horizon is 2-3 s, which meets the stated use case
  (1-2 s).

## 2. The pipeline

Inputs: a scenario's battles and its unit types. Everything else is a fixed rule
(`src/sindy_wm/models/pipeline.py`, `configs/generalisation_pipeline.yaml`):

| step | rule |
|---|---|
| states | per side and unit type: health (ten full units of the type = 1), units alive, units in reach; centroid distance; formation depth and width; each type's front offset |
| reach | per attacker / target pair: min(attack range + both radii, **6**), where 6 is SMAClite's agent target range (`docs/mixed/PROGRESS.md`) |
| rates | Savitzky-Golay, window 9, per battle |
| health | STLSQ (threshold 0.15): H_t' = Σ_a k(a→t) · R_a · σ(a→t). Candidates also include the cross terms, alive counts and a constant |
| target choice σ | proportional to the target side's type shares; no fitted numbers; σ = 1 with one type |
| withdraw, firing | rule: Blue's reach counts R·(1 − wd) (withdrawing units don't shoot, as in the scripts) |
| alive | A_t' = p_t · Ĥ_t' · A_t / H_t (+ a focus term for Red), least squares |
| distance | STLSQ (0.05) on the advance and hold switches; withdraw by the scripts' rule (danger·wd, walk_R·wd) with the scripts' withdraw radius |
| closures | reach per type: derived share at ρ, shifted by the front offset, start offset decaying in 1 s; movement gates for advance and hold by curve_fit |

## 3. How it was tested

| step | what | outcome |
|---|---|---|
| WP1-3 | one code path, one state builder, all scenarios | done |
| WP2 | pipeline on marines vs A*-2k | health and distance coefficients identical; Red's alive p +4% (explained) |
| decision rules | withdraw and target choice, written down before comparing options | committed (`c2a8c9e`) |
| WP4-5 | val table, rules applied, config frozen | withdraw (firing): rule, on a tie; target choice: proportional |
| pre-registration | expectations E1-E6, test script | committed (`c6b8e2c`) before any held-out read |
| dry run on val | test script on val | marauder physics failed: a fragile withdraw gate (closing speed 12.8 on 500 battles) |
| amendment | withdraw movement by rule instead of a fitted gate (Knut's choice) | committed (`c135280`) before any held-out read |
| second dry run | found a script bug (checked a term that no longer exists) | fixed (`7030381`), logged |
| WP6 | held-out test, run once | 19 of 20 met |

## 4. Results

### 4.1 Held-out prediction (models fitted on train)

| held-out | health 2 s (val) | health 4 s | margin R² 2 s | battle nRMSE | horizon | vs persistence / Lanchester / team at 2 s |
|---|---|---|---|---|---|---|
| marines `test4` (1000) | 0.0275 (0.0277) | 0.051 | 0.40 | 0.129 | 2 s | better / better / — |
| marauders `test` (500) | 0.0199 (0.0203) | 0.037 | 0.37 | 0.134 | 3 s | better / better / — |
| mixed `test3` (1000) | 0.0343 (0.0346) | 0.065 | 0.42 | 0.131 | 3 s | better / better / better |
| uneven `exp10` (1000, never seen) | 0.0378 | 0.070 | 0.39 | 0.122 | 3 s | better / better / better |

Health is in ten units of the scenario's first type. "Better" means a paired 95% interval entirely below 0, e.g.
mixed vs team-level at 2 s: [−0.0044, −0.0037].

### 4.2 What the pipeline discovers (refit on each held-out set)

| | firepower / game rate | advance / 6.3 | hold far / 3.15 | hold in contact / 6.3 |
|---|---|---|---|---|
| marines | 0.82, 0.86 | 0.95 | 1.02 | 0.93 |
| marauders | 0.92, 0.93 | 0.91 | 1.04 | 0.81 |
| mixed | 0.83-0.94 (8 terms) | 0.86 | 0.93 | 0.83 |
| uneven | 0.77-1.02 (8 terms) | 0.89 | 0.97 | 0.87 |

### 4.3 The frozen models (train)

The full equations are in `06_frozen_equations.md`. In short:

- **Marines:** k_B = 0.177, k_R = 0.186; p 0.66 / 0.56.
- **Marauders:** k = 0.125 / 0.126; p 0.61 / 0.54.
- **Mixed:** eight damage terms 0.060-0.189; p 0.38-0.58.
- **Distance:** advance closes at 5.4-6.0 and holding before contact at 3.1-3.4 in every scenario.
- **Withdraw:** effective rates 0.58-0.92 (danger) and 0.34-0.49 (walk_R).

## 5. What helped, what didn't

**Helped**

1. **Counting reach at SMAClite's real firing limit (6).** This turned odd marauder coefficients (57-64% of the
   game) into readable ones (90-93%). It also removed the Blue / Red asymmetry in mixed battles, and made the
   marine model transfer to marauders.
2. **Encoding the scripts' behaviour as rules instead of fitting it.** Withdraw firing as R·(1 − wd), and
   withdraw movement by danger / walk_R. The fitted versions either disappeared after the reach cap or failed to
   converge on small sets (2 of 9 refits).
3. **Per-type states in mixed armies.** They are better than a team-level model in rollouts (health 2 s −10%,
   margin R² 4 s 0.19 vs 0.08 on test3), because the composition changes during a battle.

**Did not help, or didn't work**

4. **A fitted target-choice closure:** no better than the proportional rule in rollouts, and it adds 12 numbers.
5. **Fitted withdraw switches (gates):** fragile on small data sets.
6. **Reading the withdraw movement coefficients as walking speeds:** they are effective rates, because Red chases
   withdrawing Blue units.

## 6. Caveats

- **The pre-registration was amended once,** before any held-out read, after a val dry run (withdraw movement).
  That design choice was informed by val.
- **The withdraw firing choice was a tie** on the pre-set rule, decided by one near-identical comparison.
- **The physics check is about effective rates.** Firepower per unit *in reach* is the game rate times the share
  of units in reach that fire (about 0.8-0.9 with the cap). The per-firing-unit rate is not identifiable when
  reach is also a candidate (`docs/mixed/PROGRESS.md`, checkpoint 2).
- **Weak parts:**
  - the per-type alive law (p 0.38-0.58, alive error no better than the team-level model);
  - the shared distance and the advance / hold gates (hold in contact 0.81-0.93 of the game);
  - Red's marine reach closure.
- **The uneven scenario is lopsided in strength,** so its winner accuracy (≈ 0.9 for every model) says little.
- **Not compared here:** the MLP black box. On marines it was clearly better (health 4 s 0.033 vs 0.053).
- **All four held-out sets are now spent.**

## 7. Open questions

- Why does the uneven refit put marine → marauder firepower slightly above the game rate (1.02)?
- A per-type alive law that does better than the team-level one.
- Does the reach cap of 6 also explain the stalker transfer loss of the marine model? (Likely, untested.)
- Red's intent is never observed. Inferring a hidden mode from the state (Knut's d' idea) is the natural tool for that.

## 8. Reproducing

```bash
uv run scripts/build_typed_states.py exp06_intents MARINE train val
```

(The same for exp09 and exp08.) Then:

```bash
uv run --extra explore experiments/generalisation/frozen_models.py
```

The held-out test (`heldout_test.py`) has been run; rerunning it would only reproduce the numbers above.
