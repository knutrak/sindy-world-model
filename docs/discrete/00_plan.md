# Continuous or discrete time: should the pipeline switch? (plan, 25 Nov 2026)

**Status: not run. Knut decided on 25 Nov to keep the continuous models. Draft by Claude.** It follows `docs/checks/discrete_time.md`
(exploratory, val): on raw one-step differences (D1) the health error at 2 s was 2% (marines) and 3% (mixed) lower than the
frozen continuous models' (C), with the same terms; margin R² at 2 s was slightly lower; with the relative threshold a true
firepower term was lost in mixed.

## Read this first: what is already known against D1

The held-out test's physics rule (E2, `generalisation/03`) asks every advance / hold walking speed to be within 20% of the
game's. In `discrete_time.md` (fitted on train):

| coefficient / game | C marines | D1 marines | C mixed | D1 mixed |
|---|---|---|---|---|
| advance (6.3) | 0.95 | 0.89 | 0.86 | **0.64** |
| hold, before contact (3.15) | 1.01 | 1.01 | 0.98 | 0.96 |
| hold, in contact (6.3) | 0.94 | 0.89 | 0.84 | **0.67** |

So D1 **already misses E2 in mixed battles**. The distance equation below is fitted exactly as in that run, so it will miss
again. Knut decides before the run whether S4b (below) counts. If it counts, the answer is "do not switch", and stage 1 is
only a robustness check.

## Models

| | rates | selection cutoff |
|---|---|---|
| C | frozen: SG-9 (marines, marauders, mixed); the exp12 val-run model (triple) | 0.15 × spread of the smoothed rate |
| C5 | SG-5 (5 steps = one marine volley, 0.625 s): less smoothing, still continuous | as C |
| D | raw differences (x[k+1] - x[k]) / 0.125, regressors at k | **health: 0.20 × spread of the full least-squares fit**, no smoothed data |

D's cutoff: STLSQ first fits all candidates; the cutoff is 0.20 × the spread of that fit's prediction. Shot noise inflates the
raw rate's spread but not the fit's, so no smoothed data is needed. 0.20 is fixed now; 0.15 and 0.25 are reported as
sensitivity. Distance uses 0.05 with the same rule. Closures, rules and libraries are the pipeline's. C5 separates "discrete"
from "less smoothing".

## Data

Train to fit, val to score: marines (exp06), marauders (exp09), mixed (exp08), triple (exp12). No held-out set is read and no
battle is collected. Val has been used for many choices, so this is exploratory; only stage 2 is a test.

## Expectations

**Knut's, written before the run:** …

**Rules for D (proposed; change before approval). Stage 1 passes if all that count are met:**

- **S1 terms:** D keeps exactly C's health terms in all four scenarios, and at most 6 distance terms, at 0.20; and at 0.15
  and 0.25 in at least three of the four.
- **S2 health error, 2 s:** paired 95% interval of D - C below 0 in at least three scenarios, and not above 0 in the fourth.
- **S3 margin, 2 s:** the paired 95% interval of (D - C) in the squared error of the margin change is not above 0 in any
  scenario (known: margin R² 0.394 against 0.401 in marines, 0.364 against 0.374 in mixed).
- **S4a firepower:** every firepower term 0.75-1.0 × the game rate (E2, T2).
- **S4b walking speeds:** advance, hold far and hold in contact within 20% of the game (E2), except triple (reported only, as
  in T). **Already missed by D1 in mixed.**
- **S5 not just smoothing:** D's health error at 2 s is below C5's (paired interval below 0) in at least two scenarios. If
  not, the gain belongs to less smoothing, and the fix would be C5, not discrete time.

Bootstrap over battles, 1000 resamples. Reported, not judged: 1 and 4 s, alive and distance errors, stability of the terms in
60 bootstrap refits for triple (C: 41 of 60).

## What each outcome means

| outcome | reading |
|---|---|
| all that count pass | stage 2: one pre-registered test on new held-out marines and mixed battles (1000 each) |
| S5 fails | the derivative estimate matters, not the time formulation |
| S1 fails | discrete selection needs a tuned threshold; robustness check only |
| S2 or S3 fails | no switch |
| S4b counts and fails | no switch; report the trade-off |

**A switch (stage 3)** would also mean refitting the frozen models, the interface files, the three-type and property
experiments and the planning tests, with their held-out parts redone on new sets. It is not started without a separate decision.

## Run time

The earlier run took 1 min for two scenarios. Four scenarios, two more models and the bootstrap refits: about 5-10 min,
estimated. New held-out sets (stage 2 only): about 10-15 min of collection.

## Stopping rules

Run once. No other window, cutoff, step or scenario afterwards. The continuous models stay the report's models unless stages 1
and 2 both pass and Knut decides to switch.
