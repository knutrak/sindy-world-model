# Compartments discovered, not bolted on: pre-registration (draft, 21 Nov 2026)

**Status: approved 21 Nov (E1-E4 as recommended, with the two-level selection). E5, Knut's own expectation, was
not written before approval. Written while the closed-loop test of the bolted-on variant (`13`, H5) is
running, after its train, val and counterfactual results (H1-H4) were read, and before anything of this variant is
computed. Once approved, changes are dated amendments at the end, written before the results they could affect are
read.**

## Why

The bolted-on variant (`13`) fits the compartment flows well on train but does not predict alive better in
rollouts (H3: worse in S1 at 2-4 s, no difference in S2) and does not change the picks (H4). Two reasons are
plausible:

1. **The fit does not match the use.** Its laws were fitted on 1 s window totals from measured states, with
   d-hat at the window's start, and applied every 0.125 s to the model's own predicted states. The frozen alive
   law is fitted the way it is used: on step rates, with the errors-in-variables fix (A2: the regressor is the
   model's predicted health loss).
2. **The rest of the model was held fixed.** Knut (21 Nov): if the compartment states carry meaning, discovery
   should find where, in every equation, not only where they were placed.

This variant changes both: compartment candidates go into every equation's library, the whole pipeline is
refitted on train, and the compartment laws are fitted on step rates like the rest.

## The model

The pipeline (`configs/generalisation_pipeline.yaml`, `src/sindy_wm/models/pipeline.py`) refitted on each
scenario's train set, with the same settings (thresholds, reach cap 6, withdraw rules, target rule, Savitzky-Golay
window 9), plus the compartment states W and H_W per side and type (relative wounded, as `13`).

**Selection: sparse on the state; the controls may be dense but need not be (Knut, 21 Nov).** Every equation is
fitted by a two-level STLSQ:

1. **State level (groups).** Each existing candidate term is its own group (so for them this is the pipeline's
   ordinary STLSQ); each new compartment candidate is a group made of its base term and its control interactions
   focus, hold, wd. A group stays while the norm of its standardised coefficients is at least the threshold:
   0.15 for health, 0.05 for distance and the compartment laws (the pipeline's).
2. **Control level (inside a kept group).** A control interaction is dropped when its standardised coefficient
   is below **half** that threshold (0.075 / 0.025); the group's base term is never dropped at this level.
   So the controls are pruned only when they add almost nothing, more leniently than the state.

Both levels iterate together until the kept set is stable.

| equation | existing candidates | new compartment candidates (each × 1, focus, hold, wd) |
|---|---|---|
| health H_{s,t}' | the pipeline's (R·s per attacker type, cross terms, alive counts, constant) | R_{o,a}·σ_{a,t}·w_{o,a} (do wounded attackers fire less?), R_{o,a}·σ_{a,t}·w_{s,t} (is fire drawn to wounded targets?) |
| distance d' | the pipeline's movement library | w_blue, w_red (do wounded sides move differently?) |
| alive: I, K_W, K_F, σ | — | the four laws of `13`, same groups and always-kept terms |
| reach closure, target shares, gates | refitted by the pipeline as before | none (their inputs do not involve the compartments) |

w = W / A per side and type; o = the attacking side, a its types; the other symbols as in `13`.

**Fitting the compartment laws on step rates.** From the snapshots, per step and side and type: units that cross
into wounded, deaths of wounded units, deaths of healthy units, health lost in total and by the wounded. These
event counts are smoothed with the same Savitzky-Golay filter (window 9) and divided by the step, giving rates
of I, K_W, K_F and the damage share σ. The regressors use d-hat from the *refitted* health equation on measured
reach (the A2 errors-in-variables fix), as the frozen alive law does. Rows: both sides alive, as the pipeline's.

Bookkeeping as in `13`: A' = −(K_W + K_F), W' = I − K_W, H_W' = −σ·d̂ + I·(0.5·h̄₀), with the same clipping.

## Arms

| arm | what |
|---|---|
| frozen | the frozen pipeline |
| bolted-on | the `13` variant (frozen pipeline + window-fitted compartment laws) |
| **discovered** | this variant (the primary arm) |
| discovered, no H_W | the same without the m_F and m_W groups (secondary) |

## Hypotheses and tests (fixed now)

| | hypothesis | data | test |
|---|---|---|---|
| D1 | the health equation does **not** select compartment terms; distance selects at most small ones | train | which compartment groups are kept, with their standardised size |
| D2 | the step-rate laws fit A' and W' | train, 5-fold CV by battle | CV R² of the rates of A and W from the four laws, against the frozen alive law for A' |
| D3 | rollouts predict alive better than the frozen and the bolted-on model, without making health or the margin worse | **val** (S1, S2) | alive, health and margin errors at 1, 2, 4 s, paired by start point, bootstrap by battle; both sides started at ≥ 80% as the main case, all starts alongside |
| D4 | rollouts rank intents at least as well | **cf** (S1, S2) | regret at 1, 2, 4, 8 s and the picks' eventual wins, against frozen and bolted-on |
| D5 | a planner with it wins at least as often | **fresh seeds 17 x00 000 +** (S1-S5), 400 each | wins, discovered planner − frozen planner and − bolted-on planner (horizon 3 s, commit 2 s); all three against Phase A's best fixed intent |

D5 arms per scenario: the frozen, bolted-on and discovered planners, the six fixed intents and random switching, on
the same 400 seeds. Main scenarios S1, S2, S5; S3 and S4 flagged. Read-out per scenario: beats / ties / loses
on the 95% interval, no pooled claim. Before D3-D5 are read: a test that the live state and predictions match
the recorded tables and the rollout (as for `13`), and a 5-seed smoke run on seeds 19 950 000 +.

**D1 is read as follows.** In SMAClite a unit's damage does not depend on its health. A health equation that keeps
a compartment term is therefore a warning: the term is likely absorbing another error, or the scripts react to
health in a way not known yet. Either way it is reported and looked into before D3-D5 are read as evidence for
the variant.

## Expectations (written before)

- D1: no compartment terms in health; at most a small w term in distance (withdraw radius effects).
- D2: A' as good as or better than the frozen law; W' predictable but noisier.
- D3: better than bolted-on at 2-4 s (the fit now matches the use); against frozen a small gain or a tie.
- D4, D5: a tie with the frozen planner is the most likely outcome; S1 is where a change would show first.

**Knut's expectation, including what the health equation will do:** *(to fill in before approval)*

## What is not allowed

Changing the libraries, the thresholds, the wounded definition or the arms after looking at val, cf or the fresh
seeds. Tuning on the D5 seeds. Held-out sets stay unused (spent). The frozen pipeline stays the tested model of
RQ1-RQ4; this variant is exploratory and labelled so.

## Outputs

`docs/planning/16_compartment_discovered.md` (results); `experiments/planning/compartment_discovered.py`; per-step
event tables in `data/processed/<experiment>_compart/` (new columns); `data/processed/planning/discovered_<S>.csv`.

## Decisions for Knut

| | question | my recommendation |
|---|---|---|
| E1 | compartment candidates in every equation, refit of the whole pipeline on train | yes |
| E2 | compartment laws fitted on step rates with the A2 fix | yes |
| E3 | two-level STLSQ everywhere: state groups at the pipeline's thresholds, control interactions inside kept groups at half of it; existing terms as single groups | yes (keeps the pipeline's selection for them; controls kept unless negligible) |
| E4 | D1 as a gate: a compartment term in health is looked into before D3-D5 count | yes |
| E5 | your expectation, and your answer to the health-equation question | write it before approving |

## Amendments

- **21 Nov, after D1 and D2, before anything of D3-D5 was computed.** D2's first computation compared the frozen
  alive law on rows the pipeline itself leaves out (health below 0.03, where A/H blows up) and against a different
  target than the laws (event rates vs d_alive). Fixed: both are scored on the pipeline's alive-law rows and
  against the same target, the smoothed alive rate d_alive (units/s). The fit itself was not changed.
- **D1 gate:** no compartment term entered any health equation (S1, S2); distance kept one group, w_blue
  (+0.17, +0.70·wd in S1; +0.17, +0.68·wd in S2). The gate passes: D3-D5 count as planned.
- **21 Nov, before D3 and D4 were computed.** D3 and D4 run as planned. D5 (closed loop) is deferred: Knut decides
  after D4 whether it runs. If it doesn't, it is reported as not run, with the reason.
- **21 Nov, after D4: D5 not run (Knut's decision).** Reason: in D4 the discovered model picks a different intent
  than the frozen model in 41-69% of states, but its regret and its picks' eventual wins are essentially the same
  (within ±0.004 regret, ±1 win point), so a closed loop would most likely tie; the hour is better spent elsewhere.
