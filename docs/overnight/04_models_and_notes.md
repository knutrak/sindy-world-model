# B1 and I7 written out, and what the results say (26 Nov 2026)

Notes from going through the overnight results with Knut. Nothing new was run; every number comes from `finalists.json`,
`02_blind_test.md`, `docs/single/01_results.md`, `docs/checks/README.md`, `docs/mixed/03_oracle.md` and `docs/planning/51`.

## What "single model" means

Before: one model per army, each with its own firepower number η, which absorbs damage, armour, health and cooldown. A new
unit type needs new battles and a refit. Now: one model for all armies. What the game says about a unit goes in as an input
(rate, reach, speed); only η, the alive exponent, the closure parameters and the distance coefficients are fitted, shared by
all armies.

`rate(a → t)` belongs to the attacker-target pair and comes from the stats, not from a fit (`param.py:50`):
rate = 1 / (shots_to_kill(a, t) · cooldown_a), relative to a marine on a marine; shots_to_kill = shield / damage +
health / max(damage + armoured bonus − armour, 0.5).

Range enters in two places: the share in reach (each type's own reach ρ_a) and the length scale of the distance equation.
**The speed and length scales are the army's averages** (`param.py:109`: dscale = 5.75 / mean reach, v̄ = mean speed /
3.15), not per unit. Only the share in reach uses each type's own reach.

## B1 (10 fitted terms, "B1 all")

Health, each side s and target type t (o is the other side):

    H_s,t' = −η_s · σ_s,t · Σ_a rate(a → t) · R_o,a
    η_Blue 0.1775, η_Red 0.1839
    σ_s,t = A_s,t / Σ A_s                 incoming fire split by alive units per type
    R_o,a = A_o,a · share_o,a             shooters in reach, × (1 − wd) when the shooters are Blue

Alive follows health: A_s,t' / A_s,t = p_s · H_s,t' / H_s,t, with p_Blue 0.650 and p_Red 0.557 + 0.215 · focus.

Share in reach (fitted closure plus the start error fading in τ = 1 s):

    share = Φ((ρ_a − d_eff + β · depth_o · √(2 ln n_o)) / √(0.05² + (g · depth_s)²))
    d_eff = √((d − offset_a)² + λ (width_s − width_o)²)
    (λ, β, g): Blue (0.938, 0.702, 0.992), Red (1.386, 0.531, 1.206)

Distance (formation held fixed; gate lengths × 5.75 / mean reach):

    d' = v̄ · (−0.402 walk_r·wd + 0.747 danger·wd − 5.013 close·adv − 2.770 close·hold·far − 4.693 close·hold·contact)

## I7 (11 fitted terms) = B1 plus

- Saturation: H' gets the factor sat_s = 1 / (1 + k · Σ_a R_o,a / Σ_t A_s,t), k = 0.0729 (ten shooters on one target lose
  about 42% of their effect). η rises to 0.1887 / 0.1978 to make up for it; p_Blue 0.651, p_Red 0.557 + 0.222 · focus.
- Share sees lengths in units of each type's own reach; the start error fades in 2 s / (dscale · v̄).
- Closing speed: d' = d'_eq + (d'_measured(0) − d'_eq(0)) · e^(−t / 0.125 s). The fade is one model step, so this state does
  little more than set the first step.
- Distance coefficients, gates and share parameters are the same as B1's.

## Distance did not improve

2 s distance error in the blind test (own model B0 / B1 all / I7 all / B1 left out / I7 left out):

| army | B0 | B1 all | I7 all | B1 LO | I7 LO |
|---|---|---|---|---|---|
| marines | 0.364 | 0.361 | 0.355 | 0.364 | 0.359 |
| marauders | 0.453 | 0.465 | 0.457 | 0.522 | 0.510 |
| stalkers | 0.426 | 0.535 | 0.523 | 0.621 | 0.604 |
| mixed | 0.433 | 0.433 | 0.425 | 0.471 | 0.461 |
| triple | 0.439 | 0.450 | 0.443 | 0.498 | 0.487 |

Health is shared well across unit types, movement is not: one marine-shaped distance equation scaled by mean speed and reach
does not fit stalkers (23–26% worse in the pool, about 45% when left out). The search scored health only.

## Where the error sits (earlier checks, marine model A, same closure and alive law as B1)

- **Share in reach is the largest measured error for prediction.** Measured share instead of the closure: health error
  0.029 → 0.021 at 2 s, 0.055 → 0.032 at 4 s, margin R² at 4 s 0.07 → 0.64. The closure has +0.66 Blue units too many in
  reach at 4 s, and a k fitted on the model's reach is biased low (0.161 against 0.183).
- **Alive is weak but never isolated.** One-step R² of the alive rate per type 0.05–0.16 (mixed); alive exponent variants
  (block F) gave no reliable gain. No run with measured alive exists.
- **For planning, geometry is not enough.** True reach from game copies + SINDy damage still loses to the best fixed intent
  in 4 of 5 scenarios; scores correlate 0.96 with the truth but agree with the perfect model only 56%. Close calls flip on
  small damage errors.

Open check (not run, val only, no spent data): roll out B1 with measured share, measured alive, and both, to split its
error into share, alive and damage.
