# Marines and marauders: progress log (branch `mixed-marauders`, from 19 Nov 2026)

Checklist approved by Knut on 19 Nov (with step 1b, marauders alone). Exploration phase: nothing here
is pre-registered unless stated. Data: `exp08_mixed` (marines + marauders, same mix on both sides) and
`exp09_marauders` (marauders only). The marine-only pipeline and its data are untouched.

## Step 1: scenario and collection code (19 Nov)

- `make_env(..., unit_types=(blue_types, red_types))`: each side's units as an ordered list of
  SMAClite type names. SMAClite fills its square formation grid row by row in that order.
- `MixedBattleSampler`: `CloseBattleSampler`'s draws unchanged (same stream), then from `[seed, 4]` one
  marauder share from {1/4, 1/3, 1/2} for both sides (rounded per side, at least one of each type) and a
  random order of each side's units. Recorded per episode as `n_marine_*`, `n_marauder_*`.
- Scenarios `mixed_m_mrd` and `10mrd_vs_10mrd` (not `10M_vs_10M`: macOS file names ignore case and
  would clash with `10m_vs_10m`).
- Withdraw radius: 8.0 in mixed battles (marauder reach against a marauder is 7.125, so 7 was inside it);
  7.0 × 7.125 / 5.75 = 8.67 for marauders alone, the same reach scaling as the stalker set, so the
  scaled marine model's withdraw gate matches.
- Battle limit 90 s (marauders kill slower: about 7 s per kill against 4.7 s for marines).
