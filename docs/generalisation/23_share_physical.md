# Share in reach as a state, physical library selected by STLSQ: results (26 Nov 2026)

Plan: `22_share_physical_plan.md`. Script: `experiments/generalisation/share_physical.py run` (42 s). Fitted on all train
battles (exp06, exp08), rolled out on all val battles. Output: `data/processed/generalisation/share_physical/`.
Exploratory; val has been read before.

## What STLSQ keeps

Every equation, at threshold 0.05 and 0.15 alike, keeps one term: **follow the closure**.

| scenario | equation | x' = | one-step R² |
|---|---|---|---|
| marines | Blue marine | 0.976 clo' | 0.78 |
| marines | Red marine | 0.905 clo' | 0.69 |
| mixed | Blue marine | 0.950 clo' | 0.63 |
| mixed | Blue marauder | 0.970 clo' | 0.63 |
| mixed | Red marine | 0.858 clo' | 0.46 |
| mixed | Red marauder | 0.999 clo' | 0.68 |

No term was dropped by the sign rule. With x' ≈ clo', the share is the closure plus the measured start offset, and the
offset **never fades** (the frozen model lets it fade in 1 s).

Unthresholded least squares (1,500 train battles per scenario), effect sizes in standard deviations of x':
follow +0.66 to +0.88; relax −0.01 to −0.04 (wrong sign, about zero); deaths −0.04 to +0.04 (about zero, mixed signs);
approach +0.00 to +0.05 and recede −0.01 to −0.05 (right signs, small); withdraw −0.01 to −0.05 (right sign, small).
The physical terms are there with mostly the right signs, at 1-5% of the size of follow, and one-step rates do not
separate them from noise.

## Val rollouts (health error: mean absolute over sides, as e_H)

| scenario | h | frozen health | physical health | ratio | margin R² frozen → physical | reach error (units) | margin − frozen 95% |
|---|---|---|---|---|---|---|---|
| marines | 1 s | 0.0164 | 0.0168 | 1.03 | 0.43 → 0.39 | 1.76 → 1.76 | [+0.0002, +0.0006] |
| marines | 2 s | 0.0277 | 0.0290 | 1.05 | 0.40 → 0.36 | 2.02 → 2.01 | [+0.0002, +0.0016] |
| marines | 4 s | 0.0523 | 0.0556 | 1.06 | 0.08 → 0.02 | 2.25 → 2.27 | [+0.0007, +0.0047] |
| marines | 8 s | 0.0955 | 0.1007 | 1.05 | −0.76 → −1.02 | 2.69 → 2.74 | [+0.0030, +0.0137] |
| mixed | 1 s | 0.0200 | 0.0201 | 1.00 | 0.40 → 0.41 | 1.62 → 1.54 | [−0.0006, −0.0001] |
| mixed | 2 s | 0.0346 | 0.0344 | 1.00 | 0.37 → 0.43 | 2.03 → 1.91 | [−0.0029, −0.0006] |
| mixed | 4 s | 0.0640 | 0.0633 | 0.99 | 0.15 → 0.30 | 2.14 → 2.09 | [−0.0081, −0.0018] |
| mixed | 8 s | 0.1132 | 0.1154 | 1.02 | −0.36 → −0.25 | 2.26 → 2.29 | [−0.0075, +0.0049] |

Blow-ups (share outside [−0.05, 1.05] before clipping): marines 1.8%, mixed 7.0% of start points.

## Read-outs

| rule | result |
|---|---|
| P1 signs | passes, but trivially: one term per equation |
| P2 predicts better | **fails**: marines worse at every horizon (health +3-6%, margin interval above 0). Mixed better at 1-4 s (reach and margin intervals below 0, margin R² at 4 s 0.15 → 0.30, health unchanged), not at 8 s |
| P3 stable | **fails**: 1.8% and 7.0% blow-ups (< 1% asked) |

## What it says

- With one-step rates, SINDy finds one thing about the share in reach: it moves with the closure, and the gap
  between them does not close by itself. That fits `sindyc_06` (reach has no memory given the current state): the gap
  is a lasting error of the closure for the current formation, not a transient.
- The physical mechanisms (deaths in reach, units walking in and out) are too small per step to pass any threshold.
  Discovering them would need a fit over several steps (weak form or rollout error), as S did on 16 Nov.
- Keeping the offset for good helps mixed armies up to 4 s and hurts marines. The frozen 1 s fade is a compromise
  that is safe in both; nothing here replaces it.

## Amendment 1: weak form (radius 1 s and 2 s)

`share_physical.py weak` (79 s), output in `data/processed/generalisation/share_physical/weak/`.

**Kept terms:** again only `follow`, in all 24 fits (6 equations × 2 radii × 2 thresholds): 0.78-1.02, with the one-step
R² 0.47-0.86 (higher than one-step since the windows smooth). The sign rule dropped `deaths` once (marines Red, radius
2 s, threshold 0.05). `relax` is never kept. On the 200-battle dry run, `withdraw` (−0.14 to −0.19) and `approach`
(+0.02) appeared for mixed at threshold 0.05, with right signs; on all battles they fall below the threshold.

**Val rollouts** are the same as the one-step fit to within 0.001 in health error: marines worse at every horizon
(health ratio 1.03-1.07, margin interval above 0), mixed better at 1-4 s (margin R² at 4 s 0.15 → 0.29-0.30,
intervals below 0), not at 8 s. Blow-ups 1.8% (marines) and 7.0-7.5% (mixed). P1 trivially passes, P2 and P3 fail.

**So the weak form does not change the answer.** It was meant to average out the step noise, so that slower effects
like relaxing to the closure could show; they do not, at 1 or 2 s. Together with the one-step fit and `08` (generic
library), three SINDy versions of a share state find the same thing: the share follows the closure, and the gap to
it stays. The mechanisms behind units entering and leaving reach are too weak relative to that to be selected from
team-level data, by derivatives or by integrals. Only a fit on multi-step rollout error (S, 16 Nov) has found them,
and that fit is not SINDy and gave wrong signs.

## Amendment 2: saturation in contact, own units alive, and a share without the closure

Script commands `sat`, `noclo` (about 45 s each) and `health` (51 s). Train fits, val rollouts.

### Libraries, old and new

| equation | old candidates | added |
|---|---|---|
| share x' | relax (clo − x), follow (clo'), deaths ((1 − x)(ln A)'), approach ((1 − x) max(−d', 0)/depth), recede (x max(d', 0)/depth), withdraw (x·wd, Blue) | fill (1 − x), saturate ((1 − x)·contact), contact = any unit of either side has an enemy in reach |
| share x', "no closure" | as above without relax and follow | fill, saturate |
| health H_s,t' | enemy in reach × fire split, enemy alive, constant | own units alive A_s,t (losses proportional to own numbers) |

### Health: own units alive is not selected

Unthresholded effect size of A_own −0.055 to +0.005 (threshold 0.15); every health equation is unchanged, in marines
and mixed. Losses do not scale with own numbers; units in reach already gives the near-linear decline seen in the
battles. No rollout needed.

### Share with the closure: the new terms are not selected

STLSQ keeps `follow` alone, as before (0.86-1.00); fill and saturate have effect sizes −0.03 to +0.02 when `follow` is
offered. Rollouts are those of the one-step fit above.

### Share without the closure: a sparse physical equation with the right signs

| equation | x' (threshold 0.05, after the sign rule) | one-step R² |
|---|---|---|
| marines Blue | +0.082 approach − 0.179 recede − 0.341 withdraw + 0.180 saturate | 0.20 |
| marines Red | +0.067 approach − 0.305 recede + 0.141 saturate | 0.23 |
| mixed Blue marine | +0.114 approach − 0.239 recede − 0.227 withdraw | 0.19 |
| mixed Blue marauder | +0.393 deaths + 0.109 approach − 0.188 recede − 0.164 withdraw + 0.061 saturate | 0.17 |
| mixed Red marine | +0.091 approach − 0.247 recede | 0.13 |
| mixed Red marauder | +0.117 approach − 0.285 recede | 0.23 |

`fill` (always fill up) came out negative and was dropped by the sign rule in five of six equations. Every kept term has
its expected sign. At threshold 0.15, saturate and withdraw drop out and approach + recede remain.

Val rollouts against frozen (threshold 0.05):

| scenario | h | health frozen → no closure | ratio | margin R² | reach error (units) | blow-ups |
|---|---|---|---|---|---|---|
| marines | 1 s | 0.0164 → 0.0180 | 1.10 | 0.43 → 0.39 | 1.76 → 1.82 | 0% |
| marines | 2 s | 0.0277 → 0.0325 | 1.17 | 0.40 → 0.33 | 2.02 → 2.15 | |
| marines | 4 s | 0.0523 → 0.0641 | 1.23 | 0.08 → −0.16 | 2.25 → 2.62 | |
| marines | 8 s | 0.0955 → 0.1266 | 1.33 | −0.76 → −2.30 | 2.69 → 3.88 | |
| mixed | 1 s | 0.0200 → 0.0210 | 1.05 | 0.40 → 0.40 | 1.62 → 1.57 | 2.4% |
| mixed | 2 s | 0.0346 → 0.0375 | 1.08 | 0.37 → 0.42 | 2.03 → 1.97 | |
| mixed | 4 s | 0.0640 → 0.0723 | 1.13 | 0.15 → 0.29 | 2.14 → 2.19 | |
| mixed | 8 s | 0.1132 → 0.1350 | 1.19 | −0.36 → −0.29 | 2.26 → 2.32 | |

P1 passes (all signs right, 2-5 terms). P2 fails: health error 5-33% higher, worse with the horizon. P3 passes for
marines (0 blow-ups), fails narrowly for mixed (2.4%).

### What it says

- **SINDy can find a closure-free reach equation that reads as physics:** units come into reach when the groups
  approach, leave when they move apart, Blue's withdrawing units leave, and in contact the rest fill in. Signs right,
  2-5 terms, stable for marines.
- **It predicts worse than the hand-built closure,** and more so at longer horizons. The closure ties the share to the
  current distance and formation at every step; the free equation lets errors build up.
- **Saturation is slow.** The fitted rate, 0.14-0.18 per second (time constant about 6 s), is far slower than the
  1.4 s seen in the average battle curve. Most of the rise after contact is units walking in (approach), not a
  separate fill-up.
- **Health:** losses proportional to own numbers (the exponential shape) are rejected in every equation.

**Correction (26 Nov, later):** the column "distance error" in the `val.json` files of these runs is Blue's units-alive
error (team-view index 2), not distance. Fixed in the script (index 4); no number in this file used that column.
