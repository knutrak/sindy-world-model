# Share in reach as a state, physical library selected by STLSQ: plan (26 Nov 2026)

Exploratory, asked by Knut: can SINDy discover more of the model if its coefficients may move away from the game's
numbers, as long as the signs are right and the model predicts better? Written before any fit.

## What has been tried (why this is new)

- `08_reach_state.md` (22 Nov): share as a state with a **generic polynomial library** (57-233 groups). Not sparse
  (54-57 of 57 kept), blow-ups in 2-22% of val starts, no gain at 4 s. Rejected.
- S (`sindyc_07`, 16 Nov): share as a state with a **hand library fitted on rollout error** (not STLSQ). Helped on
  marines (margin R² at 4 s 0.25 -> 0.49) but 2 wrong signs.
- Alive: spread state (`planning/10`) and discovered compartments (`planning/16`): no rollout gain. Not repeated.

New here: a **small library of physically motivated terms, each with an expected sign**, selected by plain STLSQ on
Savitzky-Golay rates (the pipeline's own method), marines and mixed.

## The equation

For side s and attacker type a, with the share x = R / A (units in reach / alive), clo the frozen closure's share,
d the distance to type a's front, depth_s the own group's depth (floored at 0.5):

    x' = c1 (clo - x)                        relax to the closure                          expected +
       + c2 clo'                             follow the closure                            expected +
       + c3 (1 - x) (ln A)'                  deaths happen in reach (c3 = 1 if all do)     expected +
       + c4 (1 - x) max(-d', 0) / depth_s    approaching brings rear units into reach      expected +
       + c5 x max(d', 0) / depth_s           receding takes units out of reach             expected -
       + c6 x wd                             Blue only: withdrawing units leave reach      expected -

STLSQ on standardised columns (threshold 0.05, the distance threshold; 0.15 reported too), SG-9 rates of x, ln A, clo
and d, rows where both sides have units alive, train only. **Sign rule:** a kept term with the wrong sign is dropped
and the rest refitted; both fits are reported.

Rollout: the frozen pipeline (`generalisation_frozen.pkl`) with x as a state started at the measured share, d' and
(ln A)' from the model's own previous step; everything else frozen.

## Read-outs (val; marines = exp06, mixed = exp08; horizons 1, 2, 4, 8 s)

- **P1 signs:** every kept coefficient has its expected sign (after the sign rule, report how many were dropped).
- **P2 predicts better:** reach error and margin error below frozen at 2 and 4 s, paired bootstrap by battle with the
  95% interval below 0, in both scenarios; health error at 1 s at most 5% above frozen.
- **P3 stable:** blow-ups (share outside [-0.05, 1.05] before clipping) in under 1% of starts.
- **Properties read off (no pass/fail):** c3 (the share of deaths that happen in reach), 1/c1 (relaxation time,
  frozen uses a fixed offset fading in 1 s), c4 / c5 against the walking speed.

Expectation (Claude, before fitting): P1 likely holds for c1, c3, c6; c2 may compete with c1. P2 at 2 s likely, at
4 s uncertain (S gained there, but with rollout fitting). Not pre-registered in the strict sense: val has been read
before for other models.

## Amendment 1 (26 Nov, after the one-step results in `23`, before any weak-form fit)

One-step STLSQ kept only `follow`. Same six terms, same sign rule and read-outs P1-P3, now in **weak form**: for each
window (radius 8 and 16 steps, every 8th step, all steps alive) the target is −∫φ′x and each column ∫φΘ, with φ the
bump of `reach_state.bump`. `follow` is integrated by parts (−∫φ′ clo), so no derivative of the closure is taken; the
rates inside `deaths`, `approach` and `recede` stay SG-9. Thresholds 0.05 and 0.15. Expectation (Claude): relax gets a
positive coefficient, since windows of 1-2 s average out the step noise; whether it helps in rollouts is open.

## Amendment 2 (26 Nov, after Knut's look at battles aligned at contact; before any fit with these terms)

Knut's reading of 669 single-movement marine battles: the share in reach rises from about 0.3 at contact to 1 within
about 3.5 s (like 1 − (1 − x₀) e^(−t/τ), τ ≈ 1.4 s); units alive fall close to linearly (per-battle R² 0.95 linear,
0.87 exponential). Two candidates were missing from the libraries:

- **Share:** `fill` = 1 − x (expected + or dropped) and `saturate` = (1 − x)·contact, contact = 1 when any unit of
  either side has an enemy in reach (expected +, about 1/τ ≈ 0.7 per second). Old terms kept. One-step STLSQ, same
  sign rule, read-outs P1-P3; rolled out on val against frozen.
- **Health:** own units alive of the target type, A_own (Lanchester's linear law / losses proportional to own numbers),
  added to the frozen health library; STLSQ at the frozen threshold 0.15 on train. Expected: dropped (the data look
  linear, which units in reach already gives). Rolled out only if kept.

Amendment 2b (same day, after the unthresholded fits with the new terms, before any STLSQ or rollout without the
closure): with `follow` in the library the new terms are about 0; without `follow` the physical terms appear with right
signs (approach +0.29 to +0.37, recede −0.21 to −0.36, saturate +0.11 to +0.19, withdraw −0.08 to −0.17 in effect size)
but one-step R² falls from 0.63-0.79 to 0.16-0.27. Added variant **"no closure"**: the library without `follow` and
`relax`, so the share equation has no closure in it at all; same STLSQ, sign rule, P1-P3 and rollout. Expectation
(Claude): rollouts worse than frozen at 1-2 s (start error no longer pinned to the closure), unknown at 4-8 s.
