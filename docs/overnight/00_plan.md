# Overnight: a better model with unit properties as inputs (plan, 25 Nov 2026)

**Status: approved by Knut 25 Nov 2026 as drafted; may be committed during the night. Nothing has been run yet.** Starting point: `docs/single/01_results.md`.

## Question

Can a model with unit properties as inputs predict unit types it never saw better than the small single model, and does a
better model give better planning?

**Primary measure (Claude's default, Knut to confirm): transfer** = health error at 2 s on unit types left out of the fit,
relative to B1. Secondary: health error on seen types, margin R², reach error, wins.

**Knut's expectation (25 Nov, at approval):** "more accurate planning results and slightly better prediction results".

**What would make the night a success (Knut):** not set separately; the plan was approved with transfer as the primary measure.

## Starting model (B1, small single model, frozen)

```
H_B' = -0.177·sum(R·rate)    H_R' = -0.183·sum(R·rate)    rate: game rate relative to marine on marine
p: 0.649 (Blue), 0.563 + 0.201·focus (Red)
d' = v·(-0.446·walk_r·wd + 0.732·danger·wd - 5.340·close·adv - 2.798·close·hold·far - 4.974·close·hold·contact)
distances × 5.75 / reach, speeds / 3.15; reach closure and gates pooled; formation held fixed
```

Comparisons: B0 each army's own model; B1; B2 a black box (MLP on the same inputs, transfer folds too) as the ceiling.

## Rules against fooling ourselves

1. **Search on train only.** Val is not used for choices; held-out sets already spent are not read.
2. **Score on left-out unit types.** Folds: leave out marauders; stalkers; the range variants; the fast or deadly variants
   (DP, CM, HM, SP). An army containing a left-out type is left out with it.
3. **Every variant is logged**, also failures. At most 60 variants in stage 1. The count is reported with the result.
4. **A placebo for each new state** (the same state from a random other battle at the same time step). A state counts only
   if it beats its placebo.
5. **A new state must be predicted by the model itself** (its own equation or closure). No measured value after the start.
6. Compactness: a finalist has at most 25 health and state terms, no cancelling pairs (no two terms with correlation above
   0.95 and opposite signs), no sign failures, at most 0.5% blow-ups.
7. The frozen pipeline models are not changed.

## Stages

### 0. Baselines and timing (30 min)
B0, B1, B2 on the folds. Time one fit and one fold. If a stage-1 variant takes more than 4 min, its data are subsampled.

### 1. Search (cap 4 h), in this order; each builds on the best so far only if it passes its gate
Gate: transfer error lower than the current best in at least 3 of 4 folds, with the paired 95% interval of the pooled
difference below 0.

| block | idea | what is tried |
|---|---|---|
| A | natural units | distance, depth, width and start offset over reach; gate parameters in those units; time over reach / speed |
| B | reach margins | mean and spread of each unit's margin (distance to the nearest enemy minus pair reach), over reach, with properties as inputs; own equation, at most 8 terms per side; one variant with reach from them only |
| C | formation that moves | depth and width with a sparse equation instead of held fixed |
| D | closing speed | d' as a state (Red's hidden mode): relaxation to the gate's value, rate over speed / reach |
| E | overkill | effective firepower with shots to kill: rate × f(shots), f from one fitted number, and units per target |
| F | alive law | exponent as a function of shots to kill and health per unit, instead of a constant |

Not tried again: spread, compartments, kite gates, front / rear groups, focus cost, large polynomial libraries.

### 2. Freeze (15 min)
At most 3 finalists: the best by transfer, the best under 12 terms, the best with reach margins if B passed. Refit on all
train data; save the models and **commit them with the blind-test script before any new battle is collected.** If no variant
passes a gate, the only finalist is B1 and stages 3-4 test B1 alone.

### 3. Blind test (collection about 40 min on 4 workers, estimated from exp13's 0.3 s per battle; read once)
New battles, seeds 40 x00 000 +, 1000 each: marines, marauders, stalkers, mixed, triple, and three **new unit variants, not
used before** (damage 9, cooldown 14 ticks; range 4.5, speed 2.6; hp 70, armour 2). The variants are fixed here.

- **T1 transfer (primary):** the finalist fitted without marauders, stalkers and mixed armies has lower health error at 2 s
  than B1 fitted the same way, on at least 3 of marauders, stalkers, mixed, triple (paired 95% interval below 0).
- **T2 new variants:** lower than B1 on at least 2 of 3 (same interval).
- **T3 seen types:** at most 1.05 × B0 on marines, marauders, mixed, triple.
- **T4 range:** on the range variant, at most 1.10 × a refit on 500 of its battles.
- **T5 margin:** margin R² at 2 s not below B1's in any army.
- **B1 itself:** within 1.10 × B0 on marauders and stalkers when fitted on marines only (the result found on val).

### 4. Planning (about 2-3 h, see below)
S1-S5, new seeds 50 x00 000 +, 400 battles each, five candidates, horizon 3 s, commit 2 s, re-plan 1 s (as `43`).
Policies: best fixed team intent (as named in `43`); the frozen planner; B1; finalists; the perfect model with the same settings.

- **L1:** a finalist beats the frozen planner in wins (paired 95% interval above 0) in at least 2 of 5 scenarios and loses in
  none. **L2:** it beats the best fixed intent in at least 2 of 5 and loses in none. Reported: share of the perfect model's
  lead captured, agreement with its choices, optimism of the chosen plan, intent use.
- A planner candidate needs a state from a live game; new states must be computed from the snapshot only.

**Known before starting:** with true geometry the SINDy planner still lost to the best fixed intent in four of five scenarios
(`51`), so a large planning gain is unlikely.

## Run time

| stage | estimate | basis |
|---|---|---|
| 0 | 30 min | single-model run: 2 min for five fits and 15 sets |
| 1 | up to 4 h (hard cap) | 60 variants × 4 folds |
| 2 | 15 min | |
| 3 | 45 min | exp13: 13 min for 13,000 battles on 4 workers |
| 4 | 2-3 h | `35`: 10,000 planner battles in 11.5 min; `43`: 6,000 battles with oracles in 53 min; a timing pilot first |
| **total** | **8-9 h** | stops at 08:00 whatever is done; stage 4 is dropped first |

Lid open, plugged in, `caffeinate`. New dependencies: none.

## Deliverables in the morning
`01_search.md` (every variant), `02_blind_test.md`, `03_planning.md`, and a one-page `SUMMARY.md`: rules met or not, and what
was not finished. No interpretation beyond the rules; that is Knut's.

## Before it starts
- Knut: expectation, success criterion, primary measure, approval; permission to commit at stage 2 without asking (the plan
  and the finalists must be committed before collection, which refuses uncommitted code).
- Commit `docs/single` and `experiments/single`.

## Stopping rules
One night. No second search on the blind-test sets, and no new variants after stage 2. A failed test is reported as it is.
