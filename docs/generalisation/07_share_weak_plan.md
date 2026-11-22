# Reach as a state, discovered from a generic library (plan, 22 Nov 2026)

**Status: approved by Knut 22 Nov 2026, with degree 3 added (see Amendments); decision rules fixed before any
fit.** Rewritten 22 Nov at Knut's request: the earlier draft chose from S's seven hand-built candidates and
fixed the structure (the closure as equilibrium), which is not discovery.

## Why

**The known limit.** Since 19 Nov the model's known limit is the **reach closure** (`final_exploration/REPORT.md`
§7). It shows up in:
- the 4 s margin;
- errors growing with Red's depth (`planning/24`);
- S1's early withdrawals (`planning/24`);
- kite: fewer Red units in reach, and Blue's reach overpredicted at 0.38 against 0.32 measured (`planning/31`,
  `33`).

**What's hand-built and what was discovered.** The reach closure is a fixed functional form fitted with curve_fit.
The only reach equation fitted so far (S) chose among seven hand-built mechanisms, and needed trajectory fitting.
The health equations are the contrast: STLSQ found R·s from a generic set of candidates.

**The question:** can SINDy discover how many units are in reach from a **generic library** of the battle's state
and Blue's intents, without a hand-built form? If not, that is itself the thesis's answer for reach: here, structure
had to be given.

## The state

For each side s and attacker type a, the **share in reach** is s_sa = R_sa / A_sa (units in reach over units alive),
a new state started at its measured value. Marines have 2 such equations, mixed 4. Everything else in the frozen
pipeline stays as it is: health, alive and distance equations, and target choice. Only the reach these equations
use changes, from closure + decaying offset to the integrated state.

## The library (generic, the same for every equation)

**Building blocks** for equation (s, a), all from the model's own state:

| block | what |
|---|---|
| s | this equation's share |
| d_a | centroid distance minus the type's front offset (the distance the type sees, as in the closure) |
| depth_own, depth_opp, width_own, width_opp | formation shape per side |
| A_own, A_opp | units alive: own type, opposing side in total |

**State candidates:**
- **Polynomials** of degree ≤ 2 in the 8 blocks: 1 + 8 + 36 = **45**.
- **Distance gates** Φ((d_a − c) / 0.5) at fixed centres c = 4, 5, 6, 7, 8, alone and times s: **10**. Geometry has
  thresholds, and a polynomial in d can't make one. The centres are a fixed grid, not fitted.
- **Deaths** (ln A_own)′ and s · (ln A_own)′, from the identity s = R / A: **2**. (ln A)′ is the step change at
  deaths, with no smoothing.

That is **57 state groups**.

**Intents:** every state group is multiplied by each of **{1, adv, hold, wd, focus, kite}**. That gives 342 columns
per equation (285 for marines, which have no kite data). All five intent controls are in every group, so any intent
can act on any state term.

**Library variant L+clo** (reported alongside): the closure's share clo_sa and its rate clo_sa′ are added as building
blocks. With them, the frozen model's form s′ = clo′ + (clo − s) is inside the library, but not imposed. **L-pure**
(without clo) is the main test; L+clo shows whether STLSQ rediscovers the closure when it's offered.

## Sparsity: sparse on the state, intents may be dense

Two-level STLSQ (`pipeline.two_level_stlsq`, used in `planning/15`):
- a **state group** (a state term with its six control interactions) stays while the L2 norm of its standardised
  coefficients is ≥ 0.05;
- inside a kept group, an **intent interaction** is dropped when its standardised coefficient is < 0.025.

Threshold sensitivity: 0.025 and 0.1, reported.

## Fitting methods

| | method |
|---|---|
| **M-weak** (main) | weak form: −∫φ′ s = Σ ξ ∫φ Θ, φ = (1 − s²)⁴ on windows of radius 1 s and 2 s (`notebooks/lib/weak_sindy.py`'s `bump`), stride 4 steps. Terms with a rate (clo′, (ln A)′) are integrated as ∫φ · rate: clo′ by parts, (ln A)′ as a sum over deaths |
| M-step (reference) | one-step STLSQ on SG-9 rates of s, as 18 Nov |

No trajectory fit: it only refines terms already chosen.

## Data (all exists; fit on train only)

| scenario | fit on | evaluate on |
|---|---|---|
| marines | exp06 train | exp06 val |
| mixed (kite in the library) | exp08 train + exp11 train | exp08 val, exp11 val |

No held-out set is read (all four are spent), and no new collection.

## Decision rules (fixed once approved)

| rule | passes if |
|---|---|
| R1 stable | per equation, the kept state groups at radius 1 s and 2 s overlap with Jaccard ≥ 0.75, and every kept group is kept in ≥ 90% of 30 bootstrap refits by battle |
| R2 readable | ≤ 8 state groups per equation, and val rollouts blow up (share leaves [0, 1] before clipping, or NaN) in < 1% of start points |
| R3 predicts | M-weak beats frozen on team reach error **and** margin error at 2 s and 4 s (paired 95% interval by battle below 0), in **both** marines and mixed; team health error at 1 s not worse by more than 5% |
| R4 intents | reported: which intent interactions survive, per equation. Kite: on exp11 val kite steps, the bias in Blue's reach (closure 0.38 vs measured 0.32) at least halves |
| R5 closure | L+clo only, reported: does STLSQ choose clo or clo′, and in what form |

**What follows:**
- **L-pure** passes R1-R3: reach is discovered without a hand-built form, and becomes a **candidate for pipeline v2**.
  That needs its own plan with new held-out sets.
- **Only L+clo** passes: discovered given the closure as a feature; a weaker claim, stated as such.
- **Neither:** reported as the answer for reach (structure had to be given), with what the library did find.
- The planner (RQ5, S4) is **not** part of this plan.

## Read-outs

- Equations per scenario, method and library, with the kept intent interactions.
- Term stability across thresholds, window radii and bootstrap.
- Val errors at 1, 2, 4 s (team reach, team health, margin) and margin R² at 4 s as in the final-exploration
  harness.
- Blow-up share.
- On exp11 val: errors in kite windows (as `planning/33` K2).
- One-step against weak: does the weak form change what is kept?

## Run time (estimated)

| step | time |
|---|---|
| weak rows (train, both radii) | a few minutes |
| fits | seconds each |
| bootstrap (30 × radii × equations) | about 5-15 min |
| val rollouts | minutes |

Memory: about 150 000 windows × 342 columns ≈ 0.4 GB per equation; built one equation at a time. A dry run on
200 battles, timed before the full run. No data collection.

## Code (to write after approval)

- A hook in `Pipeline.rollout` for the reach step, default unchanged. A test checks the frozen path rolls out
  identically, as for kite.
- `models/reach_state.py`: `ReachStatePipeline`, combinable with `KitePipeline` for the kite control.
- Generic library and weak rows.
- `experiments/generalisation/reach_state.py`.

## Expectations (written before any fit)

- **M-step:** keeps few groups and does poorly in rollouts, as 18 Nov.
- **M-weak, L-pure:** finds distance gates × (1 − s) style terms (move into reach when close) and a wd term for
  Blue. Rollouts are about level with frozen: team depth and width removed only 5-10% of the error in `24`, so the
  missing geometry is finer than the library's blocks. R3 for L-pure: about 15%.
- **M-weak, L+clo:** picks clo and s (a relaxation, rediscovered), and probably clo′. R3: about 35%.
- **Intents:** wd and kite are kept in Blue's equations; adv and hold weakly; focus mostly dropped (it moves units
  little).
- **R1** is the most likely failure with this many columns: correlated polynomials swap in and out.

## Open questions for Knut

| | question | draft answer |
|---|---|---|
| Q1 | Are fixed distance gates (a grid of centres) acceptable in a generic library? | yes: a fixed grid, no fitted centres; L-pure with polynomials only reported as a check |
| Q2 | Degree 2 enough? | yes for a first test; degree 3 would be 165 groups and worse stability |
| Q3 | Per type (as the pipeline) or per team? | per type |
| Q4 | Is reporting L+clo alongside a problem? | no, if L-pure is the main claim and L+clo is named as given the closure |

## Amendments (22 Nov, before any fit)

1. **Degree 2 and 3** (Knut, at approval): both are fitted and evaluated with the same rules. A pass by either
   counts, and the result names the degree. Degree 3 has C(11, 3) = 165 polynomial groups (L-pure), so
   177 state groups and 1,062 columns per equation with intents.
2. **Fixed distance gates** accepted by Knut (Q1).
3. **Stride 8 steps** between weak-form windows instead of 4, so degree 3 fits in memory (about 90 000 windows per
   mixed equation). Windows of radius 8 and 16 steps still overlap.
4. **Scaling:** columns and target are scaled by their root mean square, not their standard deviation. A
   column's own constant has standard deviation 0 in weak form (∫φ · 1 = 1 in every window), so std scaling is
   undefined for it. The thresholds 0.05 / 0.025 then apply to RMS-scaled coefficients.
5. **R4's kite bias, made exact:** on exp11 val start points whose controls are kite for the whole 2 s, the mean
   of (predicted − measured) Blue team reach at 2 s, reach-state model against frozen. Passes if its absolute
   value at least halves.
6. **Bootstrap:** 30 refits by battle, computed from weighted Gram matrices (same result as refitting on
   resampled rows).
7. **Blow-up (R2), made exact:** a start point blows up if a share goes NaN, or outside [−0.05, 1.05] before
   clipping, while its units are alive. The share is clipped to [0, 1] in rollouts, as the frozen closure is.
8. **R3 and the two radii:** R3 is evaluated for M-weak at threshold 0.05 at radius 1 s **and** 2 s, and passes for
   a library and degree only if it passes at both (R1 already asks for the same terms at both). M-step rollouts
   are reported, not ruled on. Frozen = the pipeline with its closure and decaying offset, rolled out through the
   same code with no share equation (checked equal to the frozen model).
