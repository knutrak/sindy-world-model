# How much can planning gain, and where? Headroom test (plan, 20 Nov 2026)

**Status: approved 20 Nov; Phase A run 20-21 Nov, results in `01_headroom.md`. Phase B not approved yet.**

## Why

RQ5 asks whether a world model improves an agent's decisions. That needs two things at once:

```
value of planning  =  (how much the choice of intent matters)  ×  (how well the model ranks the choices)
```

What we know:

- **Earlier closed-loop runs** (exp06 marines, 400 close battles, 10 Nov):
  - always advance/focus won 67%;
  - a planner using the SINDy model, re-planning every 1 s, won 70% (+2.8 points, 95% interval [+0.5, +5.2]);
  - an *oracle* planner that plans with copies of the real game won 94%.

  So there seems to be large headroom, and the model captures little of it. **Open since then:** is the oracle's 94%
  good *choices*, or perfect *foresight*? It plans with exact copies of the battle, randomness included.
- **Counterfactual ranking** (one decision, 4 s): the models rank intents well above random, but pick the best one less
  often than "always advance/nearest" does.

This plan answers the first factor first, with the simulator only (no world model): **in which scenarios, and for
which objectives, does choosing intents well matter enough to be worth a planner?** Then, as a separate decision,
the second factor.

## Phase A: headroom, simulator only

### Scenarios

All use the existing intents and scripts. None needs new behaviours.

| | scenario | why it might reward planning | setup |
|---|---|---|---|
| S1 | marines (as exp06) | the reference: earlier oracle 94% vs 67% | `CloseBattleSampler`, 10m_vs_10m |
| S2 | mixed, same mix both sides (as exp08) | composition changes during the battle | `MixedBattleSampler` |
| S3 | **range asymmetry:** Blue marauders vs Red marines | Blue outranges Red (6 vs 5.75 in practice), so holding at range might beat advancing | new sampler; counts chosen so battles are close, calibrated in a 200-battle pilot |
| S4 | **underdog:** mixed, Blue at 0.6-0.8 of Red's strength | winning is unlikely, so withdrawing to save units may be right | `MixedBattleSampler` with a lower strength ratio |
| S5 | uneven mixes (as exp10) | one side marauder-heavy | `UnevenMixSampler`, new seeds |

### Policies, on the same seeds per scenario

- **Each fixed intent** (6), plus **random switching** (as in training).
- **Oracle planner** (`envs.planner.OraclePlannerPolicy`):
  - re-plans every 1 s;
  - plays the candidate for 2 s, then advance/nearest to an 8 s horizon;
  - plans with copies of the real game.
- **Oracle with perturbed copies:** the same, but each copy gets small noise (unit positions ± 0.2, health ± 5%) and
  its own random seed. This separates *choosing well* from *knowing the future exactly*, the open question above.
  Its gain over the best fixed intent is the headroom a realistic model could hope to approach.

### Objectives, scored from the same games (no extra data)

| | objective | meaning |
|---|---|---|
| O1 | win | Blue wins the battle |
| O2 | final health margin | Blue health − Red health at the end |
| O3 | units preserved | Blue units alive at the end (mission: preserve the force) |
| O4 | exchange ratio | Red losses / Blue losses |

The oracles optimise the health margin (O2), as `OraclePlannerPolicy` does. For S4 we also run one oracle that
optimises O3 (units preserved), since that is S4's natural mission.

### Measures, per scenario and objective

- **Headroom:** perturbed oracle − best fixed intent, with a paired bootstrap 95% interval over seeds.
- **Foresight share:** (exact oracle − perturbed oracle) / (exact oracle − best fixed).
- **Which intents the oracle uses,** and when (e.g. withdraw early vs late).

### Size and cost

- 400 seeds per scenario × 5 scenarios.
- Fixed intents and random: 7 × 400 × 5 = 14 000 ordinary battles, about 15 minutes in parallel.
- **Oracle runs are expensive:** about 5 candidates × an 8 s rollout per second of battle, roughly 40-60 × an ordinary
  battle. 2 oracles × 2000 battles (plus 400 for the S4 survival oracle) is on the order of 100 000-150 000
  battle-equivalents.
- So we start with a timing pilot of 20 oracle battles. If the full run would take more than about 3 hours, the seed
  count drops to 200 per scenario, and this document is updated before running.
- **New data:** policy evaluation only, no training sets.

### Go / no-go for Phase B (fixed now)

A scenario goes on to Phase B if, for O1 or for its natural objective (S4: O3), **the perturbed oracle beats the
best fixed intent by at least 5 percentage points with the 95% interval above 0**.

## Phase B (outline only; needs its own approval)

In the scenarios that pass:

- **Model planner** (`envs.planner.ModelPlannerPolicy` with the frozen pipeline model) against the fixed intents and
  both oracles, on new seeds:
  - re-plans every 1 s;
  - commits for 1-2 s;
  - horizon 2-4 s, inside the model's reliable horizon.
- **Models:** no new training is needed. S1 uses the marine model; S2-S5 are compositions of marines and
  marauders, so the mixed model applies.
- **The result** is the share of the headroom the model captures. This is the evidence RQ5 needs before an LLM is
  added.

## Outputs

- `docs/planning/01_headroom.md` with the tables;
- a figure: headroom per scenario and objective;
- scripts in `experiments/planning/`.

## Decisions for Knut

| | question | my recommendation |
|---|---|---|
| P1 | The five scenarios above, or a subset? | all five; S3 and S4 are the most informative |
| P2 | The perturbation sizes (positions ± 0.2, health ± 5%) | keep; report one larger setting (± 0.5, ± 10%) as a sensitivity check if time allows |
| P3 | The go / no-go threshold of 5 points | keep |

## Updates before running (20 Nov)

- **Approved** by Knut as recommended (P1 all five scenarios, P2 ± 0.2 / ± 5%, P3 5 points).
- **S3 calibration** (Blue marauders vs Red marines, advance/nearest, 100-200 battles each): 1.5 marines per marauder →
  Blue wins 90%, 1.65 → 72%, 1.75 → 57%, 1.85 → 42%, 2.0 → 26%. Set to **1.8**.
- **Timing pilot** (20 battles, S2, 9 workers): oracle 26 s per battle (15-46 s), an ordinary battle 0.8 s. 400 oracle
  seeds per scenario would take about 3.5 h, so as planned the **oracles use 200 seeds per scenario** (about 1.8 h).
  The fixed intents and random switching keep 400 seeds. Every comparison with an oracle is paired on the oracle's
  200 seeds.
