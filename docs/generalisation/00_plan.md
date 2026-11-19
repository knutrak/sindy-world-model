# One pipeline, several scenarios: plan (draft, 20 Nov 2026)

**Status: draft for Knut's approval. Nothing below has been implemented.**

## The claim to support

> The same equation-discovery pipeline, given only a scenario's data and its units' game statistics,
> produces equations that (a) select the same kind of terms, (b) have coefficients that match the game's
> rules, and (c) predict comparably well, for marines, for marauders, and for mixed marine + marauder armies.

Generalisability is a requirement of the model, so this belongs in the thesis's main argument. Marines are
where the pipeline was developed; marauders and mixed armies are where it is tested.

## Where things stand

| scenario | data | model now | held-out data |
|---|---|---|---|
| marines (exp06) | train 6000, val 1000, cf | A\*-2k (`AliveVariantA`), reach 5.75, **fitted** withdraw term | `final`, `test2` **spent**; A\*-2k never scored on held-out data |
| marauders (exp09) | train 1500, val 500 | the same class, hand-made closures (reach 6, gate starting values scaled by hand) | `test` (500) **unread** |
| mixed (exp08) | train 6000, val 1000, cf 6 × 600 | `TypedA` (two types hard-coded) + a team model for distance, reach 6, withdraw **rule** | `test3` (1000) **unread** |

These are three code paths with different hand-made choices. That is the gap: today the claim is "a family
of similar recipes works", not "one pipeline works".

## The pipeline, written down as fixed rules

Inputs: one scenario's state tables and the SMAClite statistics of its unit types. Nothing else is chosen
per scenario.

1. **States per unit type and side:** health (in units of ten full units of that type), units alive, units
   in reach, centroid distance, formation depth and width, how far forward each type sits. One type means
   one group per side.
2. **Reach per pair:** min(attack range + both radii, 6), centre to centre. The 6 is SMAClite's
   `AGENT_TARGET_RANGE`, an environment rule applied everywhere (marines: 5.75, so unchanged).
3. **Rates:** Savitzky-Golay, window 9, per battle.
4. **Closures:**
   - reach per type: the derived share, ρ = the type's mean capped pair reach, shifted by the type's front
     offset, measured start offset decaying in 1 s;
   - movement gates: starting values from the attack reach (not the capped reach; gates use distance only);
   - target choice (only with ≥ 2 types): logistic closure.
5. **Health:** STLSQ (threshold 0.15) on measured reach × measured share of fire. With one type the share is
   1, and this is A\*-2k's health equation.
6. **Alive law:** A' = p·Ĥ'·A/H per type, with a focus term for Red (A\*-2k's form).
7. **Distance:** STLSQ on the movement library, with team-level reach.
8. **Withdraw:** one rule for all scenarios (decision D1).

## Work packages

| WP | what | done when | effort |
|---|---|---|---|
| **1. One code path** | Generalise `TypedA` to N unit types (any type names, the target-choice closure only when N ≥ 2, the distance equation inside the pipeline instead of a separate team model). One entry point `discover(scenario)` driven by a config file per scenario that holds only data paths and unit types. | The same function builds all three models. | ½ day |
| **2. Consistency test** | Run the pipeline on marines (one type). Compare with A\*-2k. | Coefficients within 2% and val metrics within 0.002 of A\*-2k, or every difference explained (e.g. withdraw handling). | 1-2 h |
| **3. States for all three** | Per-type state tables for exp06 (marines), exp09 (marauders), exp08 (mixed) from one builder, reach capped at 6. Train and val only. | Tables built; the single-type tables match the existing ones (test). | 1 h |
| **4. One evaluation table** | The same table for all three on val: selected terms; each coefficient ÷ its game value; one-step R²; rollout metrics (health, margin R², alive, distance at 1-8 s; whole battle; winner); baselines persistence, Lanchester and team-level; horizon (G6); gates G1-G5 where they apply. | One table, three scenarios side by side. | 2-3 h |
| **5. Settle the open choices on val, then freeze** | D1 (withdraw) and the target-choice closure vs the proportional rule, decided on val by a rule written down first (e.g. "the simpler option unless the other is better in all three scenarios"). Freeze the config and commit. | Frozen config committed before any held-out data is read. | 1 h |
| **6. Held-out test, pre-registered** | Collect the missing held-out sets (D2, D3). Write the expectations (below) and commit them with the script. Run once. | `01_preregistration.md` committed before the run; `02_test_results.md` after. | 2 h incl. collection |
| **7. Write-up** | Report (`docs/generalisation/REPORT.md`) and a notebook organised as claims with caveats, in the style of `final_01` / `mixed_01`. Draft thesis paragraphs on request. | Notebook runs end to end. | ½ day |

Total: roughly two working days. I stop and report at three checkpoints: after WP2 (does the unified
pipeline reproduce A\*-2k?), after WP5 (frozen config and val table), and after WP6 (held-out results).

## Draft expectations for the held-out test (to be fixed in WP6)

For each scenario, on its held-out set:

1. **Structure:** the pipeline selects the same health terms as on train (one firepower term per attacker type
   and target type, plus withdraw as decided), and no wrong signs (G2).
2. **Physics:** every firepower coefficient lies between 0.75 and 1.0 of the game rate; walking-speed terms
   within 15% of 3.15 / 6.3.
3. **Prediction:** health error at 4 s below the team-level baseline and Lanchester (paired 95% interval below
   0), and the reliable horizon (G6) at least 2 s.
4. **Mixed only:** the per-type model beats the team-level model on health at 4 s (paired 95% interval below 0).
5. **Reported, not used to judge:** winner accuracy, intent ranking.

The exact numbers are set in WP6, before any held-out data is read, and every expectation is reported as met
or not met.

## Decisions for Knut

| | question | options | my recommendation |
|---|---|---|---|
| **D1** | Withdraw handling, the same for all three | (a) rule: Blue units don't fire while withdrawing, nothing fitted; (b) a fitted withdraw term; (c) decide on val in WP5 by a rule written down first | **(c)**, with (a) preferred on ties: it is what the scripts do, and the fitted term vanishes once reach is capped |
| **D2** | Marines have no unread held-out set | collect a fresh marine test set (exp06 seeds 9 800 000+, 1000 battles, about 5 min) / use val only for marines | **collect**: otherwise marines are the one scenario without a held-out number |
| **D3** | An extra scenario the pipeline was never tuned on | e.g. uneven mixes (Blue mostly marauders, Red mostly marines; 1000 battles, about 5 min), collected now, read only in WP6 / skip | **collect**: the strongest single test of "different scenarios", and cheap |
| **D4** | The final marine model in the thesis | keep A\*-2k as is / replace it with the unified pipeline's marine model if WP2 shows only explained differences | **replace if WP2 passes**: one pipeline throughout the thesis |

## Out of scope (listed as further work)

Stalkers; a better per-type alive law; the front offset as a state; separate orders per unit type; a
no-withdraw collection; tuning the pipeline per scenario (this would defeat the claim).
