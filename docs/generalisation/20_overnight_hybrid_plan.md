# Overnight: three fixes for the unit-level hybrid model, alone and combined (plan, 23 Nov 2026)

**Status: approved by Knut 23 Nov 2026 ("start run now", after the outline in chat); rules fixed before any rollout.**
Follows `17` and `19`: the unit-level model gets damage nearly right in marines with separation, but reach is
27-55% worse than the closure, already at 1 s.

## The three fixes

| fix | what changes | discovered or given |
|---|---|---|
| **S** sequence switch | the walk/stop classifier also sees whether the unit walked in the previous step (groups "prev" and "prev × target in range", × intents); refitted on train | discovered |
| **R** SMAClite steering | each step's walking velocities pass through SMAClite's own collision avoidance (`smaclite.env.rvo2.rvo2.compute_new_velocity`, ORCA), two ticks per step as in the game, instead of the push-apart step | given (the simulator's rule) |
| **T** Red target stickiness | Red keeps its current target while it is alive and either in reach or attacking that Red unit, as attack-move does; else it picks the nearest | given (the script's rule) |

**Variants:** all 8 combinations. The base is `19`'s model, with push-apart separation; variants with R use steering
instead.

## Stages and rules (fixed now)

1. **H1/H2 for all 8 variants:**
   - marines: exp06 val, 300 battles; mixed reported, flagged;
   - the same start points and read-outs as `17`/`19`;
   - H1: team reach error ≤ 0.7 × the frozen closure's at 2 s and 4 s;
   - H2: margin error below frozen at 2 s and 4 s (paired 95% interval below 0).
2. **Selection:** among the variants that pass H1 in marines, the one with the lowest marines reach-error ratio at
   2 s. If none pass H1, the run stops here and reports all eight.
3. **H3** for the selected variant, only if it also passes H2:
   - five team-level candidates on exp06's counterfactual sets;
   - passes if regret is below the frozen model's at 2 s and 4 s (paired by seed).
4. **H4** for the selected variant, only if it passes H3:
   - S1 closed loop, the hybrid planner with five candidates (re-plan 1 s, commit 2 s, horizon 3 s), on `34`'s first
     200 S1 seeds;
   - paired with the existing μ, σ planner, best fixed (hold/focus) and `oracle5-h3` rows;
   - **"helps the planner"** if it beats the μ, σ planner (paired interval above 0);
   - the share of the matched ceiling is reported.

Every stage writes its table to `docs/generalisation/21_overnight_hybrid.md` as it finishes.

## Run time (estimated)

| step | time |
|---|---|
| code | a few hours |
| H1/H2, 8 variants | about 1-2 h (steering is Python, about 4× slower) |
| H3 / H4, if reached | minutes / about 30-60 min |

## Expectations (written before any rollout)

- **R** helps reach most (arrival in a queue).
- **T** helps Red's reach.
- **S** helps little.
- **About 40%** that some variant passes H1. If one does, about 50% that it passes H2-H3, and about 30% that it
  helps the planner.
