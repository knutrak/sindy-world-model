# Health compartments in the world model: pre-registration (draft, 21 Nov 2026)

**Status: approved 21 Nov as drafted (C1-C4 as recommended). C5, Knut's own expectation, was not written before
approval. Written after the train-only checks (`11`, `12`) and before any compartment quantity is computed on val,
the counterfactual sets or the closed-loop seeds. Changes are dated amendments at the end, written before the
results they could affect are read.**

## Why

`12_compartment_flows.md` (train, 1 s windows): with a unit counted as wounded once it is below half of its own
start health, the flow wounded → dead is well predicted from team quantities (CV R² 0.37-0.65 against 0.04-0.18
without the wounded count), and kills are better predicted than by the frozen power law even with the model's own
predicted damage. Focus does what Knut expected: it creates far fewer new wounded per unit of damage and finishes
the wounded about twice as fast (S1 Red marines: 0.43 → 0.09 and 0.25 → 0.49). The flow healthy → wounded is
the weak piece (R² 0.08-0.22), plausibly because the team state does not say how close the healthy units are to
the threshold. Hence a second new state: the health of the wounded group.

## The model: frozen pipeline plus compartments

Everything frozen stays: health, distance, reach and target closures (`06_frozen_equations.md`). The alive law
(ln A)' = p·(ln H)' is replaced. New states per side and unit type:

| state | meaning | scale |
|---|---|---|
| W | wounded units: alive and below half of their own start health | units / 10, as A |
| H_W | health of the wounded units | ten full units of the type, as H |

Derived, per side and type (h̄₀ = the type's mean start health, a fraction of full):

```
w   = W / A                                   wounded share
m_F = ((H − H_W) / (A − W)) / (0.5·h̄₀)        healthy units' mean health over the threshold (≥ 1)
m_W = (H_W / W) / (0.5·h̄₀)                    wounded units' mean health over the threshold (0-1)
ρ   = units in reach / A                      from the frozen reach closure
d̂  = −H'                                      the frozen health equation's rate (≥ 0)
```

Four laws, each fitted by group-STLSQ (sparse on the state groups, the controls dense inside a kept group,
Knut 21 Nov), controls ψ = 1, focus, hold, wd:

| law | meaning | state groups (× ψ) | always kept |
|---|---|---|---|
| I | healthy → wounded, units/s | d̂·(1−w),  d̂·(1−w)/m_F,  d̂·(1−w)·ρ,  d̂·w | d̂·(1−w) |
| K_W | wounded → dead, units/s | d̂·w,  d̂·w/m_W,  W | d̂·w |
| K_F | healthy → dead, units/s | d̂·(1−w),  d̂·(1−w)/m_F | d̂·(1−w) |
| σ | share of the health loss that lands on the wounded | w,  w²,  w·(1−m_W) | w |

Threshold **0.05** for all four (the distance threshold; these are new laws, not the health equation).

**Bookkeeping (rules, not fitted):**

```
A'   = −(K_W + K_F)
W'   = I − K_W
H_W' = −σ·d̂ + I·(0.5·h̄₀)          a newly wounded unit enters with health at the threshold
```

with clipping so that 0 ≤ W ≤ A, 0 ≤ H_W ≤ H, 0 ≤ σ ≤ 1, flows ≥ 0, and K_W = σ = 0 when W = 0. The health
equation (frozen) still gives H; A feeds back into reach and fire as before.

**Fitting:** each scenario's train set only (exp06 marines → S1 model, exp08 mixed → S2-S5 model). Targets are
counts over 1 s windows every 0.5 s (the flows of `12`) and, for σ, the measured share of health loss on the
wounded; inputs at the window's start, with d̂ from the frozen health equation on measured reach. Windows: both
sides alive, Blue's intent steady over the window, all movements (withdraw enters through ψ). In the rollout the
fitted rates are applied per step (0.125 s).

**Data:** W and H_W per side and type, built from the existing snapshots for train, val and the five
counterfactual branch sets of S1 and S2 (`scripts/build_compartment_states.py`); each unit's start health is its
health at step 0 of its battle. The live planner records every unit's start health at step 0 (the roster it
already records).

## Arms

| arm | what |
|---|---|
| frozen | the frozen pipeline |
| **compartments** | frozen + W, H_W and the four laws (the primary variant) |
| compartments + focus cost | also Blue's reach × (1 − c·focus) / (1 − c·f̄), from train (`04`) — secondary |
| compartments without H_W | W only: m_F and m_W left out of the libraries (tests the value of H_W) — secondary |

## Hypotheses and tests (fixed now)

| | hypothesis | data | test |
|---|---|---|---|
| H1 | the flows are predictable, and H_W helps the inflow | train, 5-fold CV by battle | CV R² per law; I with vs without the m_F groups |
| H2 | kills are better predicted than by the frozen law | train, 5-fold CV | CV R² of A' over 1 s windows: compartments vs the frozen p·(ln H)' (refit) |
| H3 | rollouts predict alive better | **val** (S1 exp06, S2 exp08) | mean absolute alive error at 1, 2, 4 s, compartments − frozen, paired by start point, bootstrap by battle; better = interval below 0. Reported for all starts and for sides that started at ≥ 80% health (the main case) |
| H4 | rollouts rank intents better | **cf** (S1, S2) | regret at 1, 2, 4, 8 s (as `02`) and the picks' eventual wins (as `03`), paired against frozen |
| H5 | a planner with it wins more | **fresh seeds 16 x00 000 +** (S1-S5), 400 each | wins, compartment planner − frozen planner (both horizon 3 s, commit 2 s), paired; both also against Phase A's best fixed intent |

H5 arms per scenario: the frozen planner, the compartment planner, compartments + focus cost, the six fixed intents
and random switching, all on the same 400 seeds (no oracle). Each row records the sides' start health, so H5 is
also reported for battles where both sides started at ≥ 80%. Main scenarios S1, S2, S5; S3 and S4 flagged as in
`07`. Read-out: beats / ties / loses on the 95% interval, per scenario, no pooled claim.

Implementation checks before any H3-H5 number is read: the live W and H_W equal the recorded tables on battles
played in a test (as `test_pipeline_planner.py`), and the planner's predictions equal the rollout on those rows.
A 5-seed smoke run per scenario on seeds 19 900 000 + checks timing and crashes only.

## Expectations (written before)

- H1: the outflow K_W and the share σ well predicted; the inflow I better with m_F, but still the weakest law.
- H2: yes, as in `12`.
- H3: a gain at 1-2 s; smaller at 4 s, where errors in I build up in W.
- H4, H5: uncertain. The compartments make focus's benefit sharper (finishing the wounded) and nearest's cost
  visible (more wounded, fewer kills now). That could move picks in either direction; S1, where the frozen planner
  ties or loses against hold/focus, is where a change would show first.

**Knut's expectation:** *(to fill in before approval)*

## What is not allowed

Changing the wounded definition, the states, the libraries, the thresholds or the arms after looking at val, cf
or the fresh seeds. Any rerun with other settings is reported as such. Held-out sets stay unused (spent).

## Outputs

`docs/planning/14_compartments.md` (results); `scripts/build_compartment_states.py`;
`experiments/planning/compartment_model.py`; data in `data/processed/<experiment>_compart/` and
`data/processed/planning/compart_<S>.csv`.

## Decisions for Knut

| | question | my recommendation |
|---|---|---|
| C1 | wounded = below half of a unit's own start health | yes (`12`; avoids the damaged-start artefact) |
| C2 | the four laws and their libraries above | yes |
| C3 | one threshold (0.05) for all four laws | yes |
| C4 | main case for H3 and H5: sides starting at ≥ 80% health, all starts alongside | yes |
| C5 | your own expectation (above) | write it before approving |

## Amendments

- **21 Nov, before H3-H5 were computed.** (1) The errors in H3 and the wins in H5 are team-level, so "sides that
  started at ≥ 80%" is applied per battle: both sides started at ≥ 80% of full health. (2) H2 compares the two kill
  laws on the same windows as the fits (all movements, steady intent, both sides alive), with d-hat for both; the
  contact-only, measured-damage comparison of `12` stays as it was. (3) Implementation check passed: on a live
  mixed battle the planner's W, H_W and start health equal the recorded tables, and its predictions equal the
  rollout (`tests/test_pipeline_planner.py`).
