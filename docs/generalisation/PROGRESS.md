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

## Decision rules for WP5 (fixed 20 Nov, before WP4; Knut approved)

Written before any val comparison of the options on marauders or mixed battles. (The marine numbers for the
withdraw options were seen in WP2: fitted term better on margin and battle nRMSE there.)

- **D1 withdraw (all scenarios):** fit both options (fitted term, rule) in all three scenarios on train; score on
  val. For each scenario, rank the two options on health error at 4 s and on whole-battle health nRMSE (rank 1 =
  lower). The option with the lower mean rank over the 3 scenarios × 2 metrics wins; a tie goes to the rule.
- **Target choice (mixed only):** use the proportional rule (no fitted numbers) unless the fitted closure has
  both lower health error at 4 s and lower whole-battle nRMSE on mixed val.
- Everything else is fixed by the plan (thresholds 0.15 / 0.05, SG window 9, reach cap 6, tau 1 s).

## WP4: the pipeline in three scenarios on val (`02_val_table.md` / `.json`, 21 Nov)

Every option passes G1, G2, G3a (light: 10 refits), G4 and G5 (better than every baseline on health at 4 s) in
every scenario. The reliable horizon (G6) is 2 s throughout, except mixed with the fitted withdraw term (1 s / none).

| val | health 4 s | margin R² 4 s | battle nRMSE | firepower / game rate | speeds / game |
|---|---|---|---|---|---|
| marines, wd fitted / rule | 0.0533 / 0.0534 | 0.073 / 0.029 | 0.130 / 0.134 | 0.83, 0.87 | 0.94-1.01 |
| marauders, wd fitted / rule | 0.0386 / 0.0386 | 0.094 / 0.080 | 0.138 / 0.139 | 0.92, 0.93 | 0.84-1.09 |
| mixed, wd rule, proportional | 0.0641 | 0.137 | 0.134 | 0.82-0.94 (all 8) | 0.84-0.98 |
| mixed, team-level A*-2k (baseline) | 0.0718 | 0.039 | 0.141 | — | — |
| Lanchester (marines / marauders / mixed) | 0.102 / 0.061 / 0.114 | < 0 | 0.18 / 0.17 / 0.18 | — | — |

Selected terms are the same in every scenario: one firepower term per attacker type and target type (plus the
withdraw term when fitted), five distance terms, no sign failures, no extra terms.

## WP5: decisions (rules fixed beforehand) and the frozen config

- **Withdraw: rule.** Mean rank 1.5 / 1.5, a tie, which the rule gives to the withdraw rule. **Caveat: the
  tie hinges on one near-identical comparison** (marauders, health 4 s: 0.038562 rule vs 0.038580 fitted). Had
  that one gone the other way, the fitted term would have won (1.33 vs 1.67). The fitted term is better for
  marines (margin R² 4 s 0.073 vs 0.029) and marauders on battle nRMSE; the rule is clearly better for mixed
  battles (health 4 s 0.064 vs 0.070, nRMSE 0.134 vs 0.143).
- **Target choice: proportional** (no fitted numbers): health 4 s 0.0641 vs 0.0650, nRMSE 0.1337 vs 0.1377.
- Frozen in `configs/generalisation_pipeline.yaml`. Consequence for D4: the marine model in the thesis becomes the
  pipeline's (withdraw rule), slightly behind A*-2k on margin R² 4 s (0.029 vs 0.087) and battle nRMSE (0.134
  vs 0.129); identical on health at 4 s.
