# Closed-loop test of the μ, σ planner (plan, 22 Nov 2026)

**Status: approved by Knut 22 Nov 2026, as drafted; rules fixed before any closed-loop battle.**

## Why

A world model with reach from the distribution of distance margins (μ, σ per side and type;
`generalisation/09`-`11`) ranks intents better than the frozen model on single decisions. Pre-registered M4 in `11`:
- regret lower at 1-4 s in marines and mixed;
- in mixed, lower than always kite/focus at 2 s.

That is open-loop evidence. This test asks the RQ5 question: **does that become more wins when the planner plays
whole battles?** The bar is the strongest simple rule, a fixed kite intent.

## The model (fixed now)

- μ, σ variant **L-p, weak form radius 2 s, threshold 0.05** (`11`). It has the lowest mixed regret at 2 s and 4 s
  (0.034 / 0.067). Marines use the same variant.
- **Blow-up fix:** in `11`, 12-18% of rollouts blew up. Each μ is now clipped to the [0.5%, 99.5%] range of its
  training values, and σ to [0, 99.5%]. The bounds are computed on train only. This is a change to the model, so
  gate G0 checks it before any battle.
- Health, alive, distance and target choice stay as the frozen pipeline.

## Gate G0 (existing data, before any battle)

With clipping, the chosen variant must still pass M4 on the counterfactual sets:
- regret below the frozen model's at 2 s and 4 s (paired 95% interval below 0) in marines and mixed;
- in mixed, below always kite/focus at 2 s or 4 s.

Also reported: how often a clip binds on val. **If G0 fails, the test stops** and that is reported.

## Planners and policies

| policy | candidates | model |
|---|---|---|
| **μ, σ planner** | mixed: the five + kite/nearest, kite/focus (7); marines: the five (no kite data for marines) | μ, σ variant |
| frozen planner | the five (as Phase B) | frozen pipeline |
| fixed kite/nearest, fixed kite/focus | — | — |
| Phase A's best fixed intent | — | — |

All planners re-plan every 1 s, commit 2 s, horizon 3 s, as in Phase B.

**Live states:** at each re-plan, μ and σ come from the live snapshot with the same margin code as the training
tables (`states/margins.py`). A test checks the live values equal the table values on a recorded battle.

## Seeds and size

| | seeds |
|---|---|
| main test | fresh, used nowhere: S1 21 000 000 +, S2 21 100 000 +, S3 21 200 000 +, S4 21 300 000 +, S5 21 400 000 + |
| smoke run | 19 950 000 + (5 per scenario) |

400 battles per scenario and policy, paired by seed.

## Read-outs and rules (fixed once approved)

**Best fixed kite per scenario**, from P2 on Phase A's seeds (`27`): kite/nearest in S1, S2, S4, S5; kite/focus in
S3.

| | comparison | how it is read |
|---|---|---|
| **C1 (main: S2, S4, S5)** | μ, σ planner − best fixed kite, wins | paired bootstrap 95%: beats (above 0), ties, loses (below 0) |
| C2 (all scenarios) | μ, σ planner − frozen planner | the same; isolates the model change |
| C3 (flagged: S1, S3) | μ, σ planner against best fixed kite and Phase A's best fixed | S1 has no kite candidate; S3 is already at 99% |

**RQ5 statement:** "planning with the μ, σ world model is useful in this testbed" holds if C1 **beats in at least
one main scenario and loses in none**. Otherwise planning is reported as not beating the strongest simple rule.

**Diagnostics:**
- intent use (share of re-plans per pick, kite share);
- how close the top two scores are (gap < 1 hp);
- clip binding;
- planner seconds per battle.

## Run time

Timed on the smoke run first and stated before the main run. The planners have Phase B's cost per re-plan plus the
μ, σ library (about 300 columns per equation and step). Expected total: several hours for 5 scenarios × 5 policies
× 400 battles.

## Expectations (written before any battle)

- G0 passes: clipping touches few states.
- **C1:** S2 ties (fixed kite already 88.5%); S5 ties or a small gain (kite 59%, the oracle with 7 candidates gained
  little there in `27`); S4 has about a 30% chance of beating kite (34% → oracle 79%), and also a real risk of losing,
  since S4 is outside the training range.
- **C2:** the μ, σ planner beats the frozen planner in S2, S4 and S5, mostly through using kite.

## Open questions for Knut

| | question | draft answer |
|---|---|---|
| Q1 | Run the seven-candidate oracle on the fresh seeds too (for the share of headroom captured)? | no: costly; use `27` P3's levels as a reference only |
| Q2 | L-p or L-nop? | L-p radius 2 s (best M4); L-nop reported only in G0 |
| Q3 | Fewer battles to save time (200 per scenario and policy)? | 400, as Phase B, so the results compare |
