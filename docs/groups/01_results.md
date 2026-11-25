# Front / rear group states: results (25 Nov 2026)

Plan: `00_plan.md` (with amendment 1). Script: `experiments/groups/groups.py`. Logs: `logs/*_amended.txt` (the first run
is in `logs/marines.txt`, `logs/marauders.txt`). **Exploratory, train and val only;** no held-out set was read. Fitted
on 1500 battles (4 s windows), scored on all val windows (1 per second, 8827 marines, 7777 marauders).
Health in ten units of the type, mean per side.

## Read-outs against the plan

| | result |
|---|---|
| **G1 gain** (FR-c below T at 2 and 4 s, marines) | **fails**: +0.0004 at 2 s, +0.0058 at 4 s (worse) |
| **G2 not just parameters** (FR-c below P-c) | **fails**: 2 s ties (+0.0001), 4 s worse (+0.0060) |
| **G3 structure** | **fails on the bounds**: two terms per equation, no others, no wrong signs, but 4 of 8 marine k are below 0.75 and one is 1.01 |

**By the plan: not useful.** One exception in marauders, where FR-c is better than P-c at 2 and 4 s, and than T at 2 s
only. FR-r was not in the judged comparisons.

## Rollouts

marines (health error, margin R² at 4 s; alive error in units):

| model | health 1 s / 2 s / 4 s | mR² 2 s / 4 s | alive 4 s |
|---|---|---|---|
| T (team, same windows) | 0.0165 / 0.0277 / 0.0524 | 0.389 / 0.056 | 0.918 |
| frozen | 0.0165 / 0.0277 / 0.0523 | 0.401 / 0.080 | 0.913 |
| FR-c (front/rear, fitted target choice) | 0.0162 / 0.0281 / 0.0582 | 0.356 / −0.143 | 0.970 |
| FR-r (front/rear, proportional) | 0.0164 / 0.0271 / 0.0510 | 0.440 / 0.144 | 1.038 |
| P-c (random halves, fitted) | 0.0169 / 0.0279 / 0.0522 | 0.441 / 0.113 | 0.989 |
| P-r (random halves, proportional; post hoc) | 0.0169 / 0.0276 / 0.0509 | 0.456 / 0.186 | 0.974 |

marauders:

| model | health 1 s / 2 s / 4 s | mR² 2 s / 4 s | alive 4 s |
|---|---|---|---|
| T | 0.0118 / 0.0204 / 0.0387 | 0.342 / 0.088 | 0.726 |
| frozen | 0.0118 / 0.0203 / 0.0386 | 0.344 / 0.091 | 0.725 |
| FR-c | 0.0117 / 0.0198 / 0.0382 | 0.346 / 0.033 | 0.740 |
| FR-r | 0.0118 / 0.0197 / 0.0370 | 0.388 / 0.194 | 0.842 |
| P-c | 0.0121 / 0.0204 / 0.0390 | 0.353 / 0.099 | 0.777 |
| P-r | 0.0121 / 0.0204 / 0.0387 | 0.364 / 0.148 | 0.777 |

Paired differences in health error (95% battle bootstrap), a − b:

| | marines 2 s | marines 4 s | marauders 2 s | marauders 4 s |
|---|---|---|---|---|
| FR-c − T | +0.0004 [+0.0001, +0.0006] | +0.0058 [+0.0050, +0.0067] | −0.0005 [−0.0007, −0.0004] | −0.0005 [−0.0010, 0.0000] |
| FR-c − P-c | +0.0001 [−0.0001, +0.0003] | +0.0060 [+0.0053, +0.0067] | −0.0006 [−0.0007, −0.0005] | −0.0008 [−0.0011, −0.0004] |
| FR-r − T | −0.0006 [−0.0007, −0.0004] | −0.0014 [−0.0018, −0.0011] | −0.0007 [−0.0008, −0.0005] | −0.0017 [−0.0020, −0.0014] |
| FR-r − P-r (post hoc) | −0.0005 [−0.0006, −0.0004] | +0.0001 [−0.0003, +0.0005] | −0.0007 [−0.0007, −0.0006] | −0.0017 [−0.0019, −0.0014] |
| P-r − T (post hoc) | −0.0001 [−0.0003, +0.0001] | −0.0016 [−0.0020, −0.0011] | −0.0000 [−0.0002, +0.0001] | −0.0000 [−0.0003, +0.0002] |

## What the groups show

- **The front takes most of the damage.** In 2 s it takes 81% (Blue) and 74% (Red) in marines and 84-85% in
  marauders, with about half the health. Fitted target choice knows this (a weight of 2.8-4.3 on the position difference).
- **The fitted coefficients differ by position.** Front on front: 0.95 and 1.01 of the game (marines), 1.00 and 1.03
  (marauders). A rear attacker against the enemy's front: 0.62-0.70. The rear counts as in reach but does less damage per
  unit. Random halves give the same k everywhere (0.75-0.80 and 0.83-0.86).
- **Where FR-c goes wrong.** In marines at 4 s it predicts too little health in the front (Blue −0.018) and too much in
  the rear (Blue +0.031, Red +0.040). The front's position and labels are held fixed, so it keeps drawing fire when the
  real rear has moved up. FR-r has the opposite, larger group errors (front +0.03, rear −0.03): its proportional rule ignores
  the position, and the errors partly cancel in the team sum.
- **FR-r's marine gain at 4 s is not about position:** random halves do as well (P-r ties it). At 2 s, and in marauders at
  both horizons, FR-r beats its placebo by 0.0005-0.0017 (2-4% of the error). Its alive error is 13-16% worse than T's.

## Caveats

- Labels are fixed for each window (4 s). A dynamic split would need flows between groups.
- Positions are held at the start value, so a longer horizon than 4 s was not tried.
- The smoke test (60 val battles) was looked at before the full run; the judged variant was not changed.
- One run per scenario; val reused many times.
