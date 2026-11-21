# A health-spread state for the alive law: pre-registration (21 Nov 2026)

**Status: written before any spread quantity is computed on val, the counterfactual sets or the closed-loop
seeds. Changes are dated amendments at the end, written before the results they could affect are read.**

## Why

The alive law is the pipeline's weakest part (p 0.38-0.58 across types). Knut's point (21 Nov): focus fire hardly
changes how much health a team loses, but it changes how fast units die, because concentrated damage finishes
units sooner. The frozen model carries that only as a control term in Red's alive law,

```
(ln A_R)' = (0.562 + 0.260·focus) · (ln H_R)'
```

which forgets the past: after a switch from focus to nearest, the wounded units are still there, but the law
returns at once to 0.562. Team health and units alive do not say how the health is spread over the units. Two
teams with equal H and A can be ten half-health units (no kills soon) or five full and five nearly dead units
(kills now). Hypothesis: **one state per side and unit type that describes the spread carries focus's effect
on kills, with memory, and makes the alive law better.**

## The state

Per side and unit type, over the alive units, with h_i = (hp + shield) / full of unit i:

| candidate | definition | 1 / 0 means |
|---|---|---|
| **evenness c** | (Σ h_i)² / (N · Σ h_i²) | 1: all equally healthy; low: a few healthy, many nearly dead |
| **fragile share f** | share of alive units with h_i < 0.5 | 0: nobody below half; 1: everyone below half |

With one unit or none alive, c = 1 and f = h < 0.5 for that unit (or 0). **Choice between c and f, on train
only:** the one whose alive law (below) gives the larger cross-validated R² of d_alive (5 folds by battle,
pooled over sides and types in S1 and S2). Ties within 0.005: c (simpler to reason about).

## The model: frozen pipeline plus the spread state

Everything frozen stays: health, distance, reach and target closures (`06_frozen_equations.md`). Two changes:

**Alive law.** For each side and type, with g = Ĥ'·A/H the frozen law's own regressor (the health equation's
rate, as in fitting):

```
(ln A)' = p(s, u) · (ln Ĥ)'      fitted as  d_alive = Σ_k θ_k · g · φ_k(s) · ψ_j(u)
state groups  φ(s):  1,  (1 − s),  (1 − s)²            (s = c, or 1 − f, so "1 − s" grows with spread)
controls      ψ(u):  1,  focus,  hold,  wd              (dense within a kept group)
```

**Spread dynamics.** For each side and type, with r = −(ln Ĥ)' ≥ 0 the relative rate of health loss:

```
s' = Σ_k θ_k · φ_k(s, r) · ψ_j(u)
state groups  φ:  1,  s,  r,  r·s
controls      ψ:  1,  focus,  hold,  wd,  focus·far,  hold·far        (far = the model's far gate)
```

**Sparsity on the state, not on the controls (Knut, 21 Nov).** A group-STLSQ: each state group φ_k enters with
all its control interactions or not at all. Least squares on the kept groups; a group is dropped when the norm
of its standardised coefficients is below the threshold; repeat until stable. Thresholds: **0.15 for the alive
law** (the pipeline's health threshold), **0.05 for the spread dynamics** (the distance threshold). The group
"1" in the alive law is always kept (it is the frozen law's p).

**Rollout:** s evolves per side and type (clipped to [0.05, 1]); the alive update is A·(H₁/H₀)^p(s, u), as
in the frozen rollout. s starts at its measured value.

**Fitting:** each scenario's train set only (exp06 marines → S1 model, exp08 mixed → S2-S5 model); rates by
Savitzky-Golay, window 9, as every other rate. Rows with fewer than 2 alive units of the type are left out of
the spread fit.

## Arms

| arm | what |
|---|---|
| frozen | the frozen pipeline |
| **spread** | frozen + the spread state (the primary variant) |
| spread + focus cost | also Blue's reach × (1 − c·focus) / (1 − c·f̄), from train (`04`) — secondary |

## Hypotheses and tests (fixed now)

| | hypothesis | data | test |
|---|---|---|---|
| H1 | the spread state improves the alive law | train, 5-fold CV by battle | CV R² of d_alive: spread > frozen form, per side and type |
| H2 | focus acts on kills through the spread: Red's focus term shrinks | train | Red's p(s, focus) − p(s, nearest) at the median s, vs the frozen 0.260 |
| H3 | Blue's focus lowers Red's evenness under fire | train | sign of the focus terms in Red's s' (focus·r and focus·r·s) |
| H4 | rollouts predict alive better | **val** (S1 exp06, S2 exp08) | mean absolute alive error at 2 s and 4 s, spread − frozen, paired by start point, bootstrap by battle; better = interval below 0 |
| H5 | rollouts rank intents better | **cf** (S1, S2) | regret at 1, 2, 4, 8 s (as `02`) and the picks' eventual wins (as `03`), spread vs frozen, paired bootstrap |
| H6 | a planner with it wins more | **fresh seeds** 15 x00 000 + (S1-S5), 400 each | wins, spread planner − frozen planner (both horizon 3 s, commit 2 s), paired; also both against Phase A's best fixed intent |

H6 arms per scenario: the frozen planner, the spread planner, the spread + focus-cost planner, the six fixed intents
and random switching, all on the same 400 fresh seeds (no oracle). Main scenarios for H6: S1, S2, S5; S3 and S4
flagged as in `07`. Read-out: beats / ties / loses on the 95% interval, per scenario, no pooled claim.

## Expectations (written before)

- H1-H3: likely yes. The kill counts in `03` (Check 1) say focus nearly doubles kills per unit of damage.
- H4: a small gain, mostly at 4 s, where the memory of past focus matters.
- H5, H6: little change. Judged by outcomes, focus and nearest are about even (within 1 win point), so a
  better alive law may move few decisions. A gain would most likely show in S1, where the frozen planner lost
  to hold/focus.

## What is not allowed

Changing the state definition, the libraries, the thresholds or the arms after looking at val, cf or the fresh
seeds. Any rerun with other settings is reported as such, not as the result. Held-out sets stay unused (spent).

## Outputs

`docs/planning/10_spread_state.md` (results); `scripts/build_spread_states.py`;
`experiments/planning/spread_state.py`; data in `data/processed/<experiment>_spread/` and
`data/processed/planning/spread_<S>.csv`.

## Amendments

- **21 Nov, before the closed-loop runs were read.** (1) H3 is read from the *net* effect of focus on Red's s'
  (focus minus nearest at the median contact-row c and r, train only), since the individual focus terms in a
  dense control library partly cancel; the measured train difference in Red's d_s under focus vs nearest is
  reported next to it. (2) A 5-seed smoke run per scenario on seeds 19 800 000 + checked timing and crashes only.
