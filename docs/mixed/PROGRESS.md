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

## Checkpoint 1: the exp08 pilot (300 mixed battles, `01_pilot.md`)

- Collected in 69 s. Blue wins 60%. Battles last 15-17 s, against about 9 s for marines alone, and
  get longer with more marauders (share ~1/2: 17.3 s). None hit the 90 s limit.
- On the winning side 56% of the marauders survive, against 27% of the marines: marines die first.
- **Fire allocation.** Blue's fire on marauders (0.53-0.57) follows the marauder share among Red's alive
  units (0.54-0.60), slightly below it (Blue's focus picks marines, the weakest by absolute hp). Red puts
  *more* fire on marauders than their share (0.51-0.53 against 0.40-0.47): Red's AI keeps its first
  target and takes the nearest new one, and the marauders' longer range brings them into contact first.
  So the proportional rule is close for Blue and biased for Red.
- **Preview of the damage matrix (measured fire, one-step least squares).** Marine attackers: 0.92-1.01
  of the game rate on both target types. Marauder attackers: 0.85-0.90. With measured allocation the
  matrix is recovered with the right pattern, including the low marine → marauder entry (0.063-0.065
  against 0.064). This is a preview of checkpoint 2, not the closure model.

## Step 1b: marauders alone (exp09, `02_marauders.md`)

Collected: train 1500, val 500, test 500 (not read). Marauder battles last about 17 s; Blue wins 50-54%.

| on exp09 val | health 4 s | margin R² 4 s | alive 4 s | distance 4 s | battle nRMSE |
|---|---|---|---|---|---|
| persistence | 0.140 | −0.01 | 1.77 | 1.09 | 0.572 |
| A / A\* / A\*-2k, marines scaled (no refit) | 0.057 / 0.055 / 0.054 | −0.45 / −0.59 / −0.49 | 0.87-0.96 | 0.78 | 0.153-0.160 |
| A\*-2k refit on marauders | **0.034** | **0.13** | 0.75 | 0.63 | **0.124** |

- **The refit finds the same 11 terms** as on marines: same structure, no new terms.
- **Transfer without refitting** beats persistence by far but gets the margin wrong (R² −0.45 to −0.59).
  The scaled firepower is too high: 0.115-0.121 against the refit's 0.077 / 0.086.
- **Why: the share of units in reach that fire is not unit-free.** From the snapshots (200 val battles
  each): marines in reach are firing 79% of the time, marauders only 53-54%. The refit's firepower is
  57-64% of the game rate (0.135), which matches 0.135 × 0.54 ≈ 0.073 (slightly above it). `scaled()`
  assumes the firing share carries over from marines; for marauders it doesn't. The *per-firing-unit* rate
  is close to the game's (pilot preview: 0.86-0.90).
- k_B < k_R again (0.077 vs 0.086), the direction Red's shorter acquisition radius predicts. For marauders
  the gap between Red's acquisition (5 + 0.5625 = 5.56) and the reach (7.125) is much wider than for marines.
- Distance: hold before contact 3.00 and withdraw 3.05 (walking speed 3.15); mutual advance 5.61 (6.3);
  hold in contact 3.87 (marines 5.91). Flee term larger (0.73 vs 0.30).
- Alive exponents p_B 0.63, p_R 0.53, close to the marines' (0.66, 0.54); focus term smaller (0.15 vs 0.24).

**Reading (hedged).** The structure transfers (same terms, same signs, similar alive exponents), but one
number doesn't: the share of units in reach that are actually firing. That share is where marines and
marauders differ, and it looks like the quantity a per-type mixed model will need per type. Why
marauders in reach fire so much less is not yet known (bigger bodies crowding? the wider band between
Red's acquisition radius and the reach? longer turn-around after a kill?).

## Full exp08 collection (19 Nov)

`scripts/collect_mixed.py` (commit of the script before collecting; logs in `data/logs/mixed_2026-11-19/`),
14 sets in parallel, about 8 minutes:

| set | battles | seeds | Blue wins |
|---|---|---|---|
| train_0 … train_5 | 6 × 1000 | 10 100 000 + 1000 k | 59-63% |
| val | 1000 | 10 500 000 | 60% |
| cf_<intent> (6 intents) | 6 × 600 | 10 600 000 | 59-61% |
| test3 | 1000 | 10 900 000 | 58% (**not read**; only the collector's summary line was seen) |

None truncated at 90 s. Per-type state tables in `data/processed/exp08_mixed/` (train merged with
episode ids shard × 1000 + episode; test3 not built). Checks: no missing values in train (767 077 steps,
45 columns); the six cf branches are identical at the branch step in all 599 seeds checked.

## Checkpoint 2: one-step damage matrix (`03_oracle.md`, `experiments/mixed/oracle.py`)

Fitted on exp08 train (6000 battles), one-step R² on val, STLSQ threshold 0.15, measured reach and
measured fire allocation (no closures, no rollouts).

**(ii) reach × measured allocation share (the form a rollout model uses) recovers the matrix.** STLSQ
keeps exactly the four attacker → target terms per side (plus a withdraw term for Blue's marauders),
with tight battle-bootstrap ranges (±1%):

| attacker → target (Blue / Red attacking) | game | fitted, per unit in reach | / game | / (game × firing share from 1b) |
|---|---|---|---|---|
| marine → marine | 0.213 | 0.173 / 0.170 | 0.81 / 0.80 | 1.03 / 1.01 |
| marine → marauder | 0.064 | 0.055 / 0.053 | 0.86 / 0.83 | 1.08 / 1.04 |
| marauder → marine | 0.198 | 0.141 / 0.121 | 0.71 / 0.61 | 1.32 / 1.14 |
| marauder → marauder | 0.135 | 0.093 / 0.081 | 0.69 / 0.60 | 1.27 / 1.11 |

- Marine entries are the game rate times the marines' firing share from step 1b (0.79), within 1-8%.
- Marauder entries sit above the single-type firing share (0.54): in mixed battles marauders in reach fire
  more often than in marauder-only battles (implied 0.60-0.71).
- Blue (scripted) is again more effective per unit in reach than Red (AI), now for both attacker types.
- Withdraw: Blue's marauders lose all their fire when withdrawing (+0.185 against −0.141; +0.097 against
  −0.093). No withdraw term is kept for Blue's marines: they leave their shorter reach quickly when
  withdrawing (radius 8), so R_marine already drops (hedged reading).

**(i) measured fire per firing unit is not identifiable with reach in the library.** Fire counts and reach
are nearly collinear (firing ≈ 0.8 × reach for marines), and STLSQ splits the effect between them, with a
wrong-signed cross term (+0.11 on marines' health from marine fire on marauders). Without reach as a
candidate (pilot preview) the per-firing-unit rates came out at 0.85-1.0 of the game. A SINDy caveat:
which of two collinear physical descriptions gets selected depends on the library.

**(iii) the proportional allocation rule is clearly worse.** Val R² drops from 0.41-0.62 to 0.21-0.47 and the
marine → marauder term is lost for Blue's marauders. The allocation closure matters.

**Does the per-type state help one step ahead? Barely, for total health.** R² of the total health rate
(hp/s) on val: team-level one-k fit 0.44 (Blue) / 0.56 (Red); per type with measured shares 0.46 / 0.57;
per type with the proportional rule 0.41 / 0.56. The value of the per-type state, if any, will have to show
in composition (which units die) and in rollouts, where the composition changes over the battle.

**Alive law per type** (A*-2k's form): p = 0.53 / 0.41 (Blue marines / marauders), 0.45 / 0.37 (Red),
lower than in single-type battles (0.53-0.66), with low one-step R² (0.05-0.16). Not yet understood.

**Reading (hedged).** The structure holds: with the right reach and allocation, one-step SINDy finds the
heterogeneous Lanchester matrix with the game's pattern. Two things carry the difficulty: the allocation
closure (proportional rule loses a lot) and the firing share per type (not unit-free, and different in
mixed and single-type battles). One-step total-health accuracy does not improve on the team-level model.
