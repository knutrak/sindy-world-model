# One pipeline, several scenarios: progress log

Plan: `00_plan.md` (approved 20 Nov with the recommended decisions: D1 decided on val by a rule fixed
beforehand, rule preferred on ties; D2 fresh marine test set; D3 uneven-mix scenario; D4 replace A*-2k if WP2
passes).

## Held-out sets collected (20 Nov, from commit before collection; not read)

- `exp06_intents/test4`: 1000 fresh marine battles, seeds 9 800 000+ (`collect_intents.py test4`).
- `exp10_uneven/test`: 1000 battles with uneven mixes (one side 2/3 marauders, the other 1/4; which side is
  drawn per battle), seeds 12 000 000+ (`collect_uneven.py`, `UnevenMixSampler`).
- Unread as before: `exp09_marauders/test` (500), `exp08_mixed/test3` (1000).
Only the collectors' summary lines were printed. Building their state tables needs `--held-out`.

## WP1 and WP3: one code path, one builder

- `src/sindy_wm/models/pipeline.py`: `Pipeline(types, withdraw="rule"|"fitted")` for one or two unit types;
  the distance equation is fitted inside (no separate team model). The exploratory `TypedA` is left as it was.
- `scripts/build_typed_states.py`: per-type tables for any scenario (reach capped at 6), written to
  `data/processed/<experiment>_typed/`. Built: exp06 (marines) train 6000 / val 1000, exp09 (marauders) train
  1500 / val 500, exp08 (mixed) train 6000 / val 1000.

## WP2: the pipeline on marines against A*-2k (`01_consistency.md`)

Same train battles, same val start points (checked identical to the harness's).

- **Health and distance coefficients are identical** to A*-2k (k_B −0.1770, k_R −0.1861, k_wd +0.1633; all five
  distance terms to 4 decimals), and STLSQ keeps no other health term.
- **One explained difference: Red's alive exponents** (p_R 0.562 vs 0.539, +4%; focus 0.260 vs 0.238, +9%).
  A*-2k's alive regressor uses only the main firepower term (k R_B A / H); the pipeline uses its whole predicted
  health rate, which includes the withdraw term when Blue withdraws. p_B is identical because Blue's health
  equation has no withdraw term.
- Val metrics: within 0.002 except margin R² at 4 s (0.073 vs 0.087) and battle nRMSE (+0.0015), both from the
  p_R difference above.
- With the withdraw *rule* instead of the fitted term, marines lose a little: margin R² 4 s 0.029, battle nRMSE
  0.134 (the fitted term says withdrawing marines keep about 12% of their fire, the rule says 0%).

**Verdict: pass, with one explained difference** (the alive regressor includes the withdraw term). D4: the
pipeline's marine model can replace A*-2k; its only change is in Red's alive exponents.
