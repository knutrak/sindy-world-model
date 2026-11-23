# Hybrid movement plus separation (plan, 23 Nov 2026)

**Status: approved by Knut 23 Nov 2026 ("do that hybrid movement plus separation"); criteria fixed before any
rollout.** Follows `17`: without collision avoidance, simulated units arrive together, and reach is worse than the
closure.

## The change

After each step's movement, overlapping units are pushed apart until no two alive units are closer than the sum of
their radii (SMAClite's unit sizes: marine radius 0.375, marauder 0.5625). Each pass moves both units of an
overlapping pair half the overlap apart, along the line between them; 4 passes per step. This is a known rule (no
overlap), not a fitted one. Everything else is unchanged from `16` / `17`:
- the same switch classifiers;
- speed 3.15 and heading by intent;
- damage at the frozen firepower.

## Stages and criteria

These are `16`'s, unchanged:

| stage | passes if |
|---|---|
| **H1 geometry** | team reach error ≤ 0.7 × the frozen closure's at 2 s and 4 s (marines, exp06 val, 300 battles; paired by battle) |
| H2 full model | margin error below frozen at 2 s and 4 s |
| H3 ranking | regret below frozen at 2 s and 4 s on exp06's counterfactual sets |
| H4 closed loop | S1, `34`'s first 200 seeds: "helps the planner" if it beats the μ, σ planner (paired interval above 0) |

Stages run in order, each only if the earlier ones pass. Mixed is reported alongside, flagged.

## Expectations

- **H1:** about 50%. Separation should stop the pile-up, but SMAClite's collision avoidance also steers units
  around each other, which simple pushing does not.
