# Is it the model or the setup? A black-box world model in the same planner (plan, 22 Nov 2026)

**Status: approved by Knut 22 Nov 2026, as drafted; rules fixed before any training.**

## Why

No world-model planner has beaten a fixed kite rule (`35`). Two explanations fit:

| explanation | what it predicts |
|---|---|
| **The interpretable model is the limit** (SINDy too weak for 3 s decisions) | a flexible black-box model, given the same inputs, beats fixed kite |
| **The setup is the limit** (near-ties, a 3 s horizon, one dominant per-unit rule) | a black-box planner also loses to fixed kite |

In `final_exploration` the black-box MLP was clearly better than SINDy at 4 s (margin R² 0.60 vs about 0). So this comparison separates the two explanations.

## The black-box model (fixed now)

A **typed MLP world model**, built like `final_exploration`'s `MLPWorld` (J, wide):

| | choice |
|---|---|
| step | x_{t+1} = x_t + DT · f(x_t, inputs, u_t) |
| state x | the pipeline's typed state: health and units alive per side and type, plus distance |
| constant inputs | shape (depth, width), front offsets, start share in reach per side and type, and the start **μ, σ** of the reach margins. That is the same information the μ, σ planner has at a decision |
| controls u | advance, hold, withdraw, focus, kite |
| network | 2 hidden layers of 128, tanh |
| training | rollout loss over 24 steps (3 s, the planner's horizon); 40,000 random start points; 30 epochs; Adam, cosine schedule; seed 0 |
| outputs | clipped as in `MLPWorld`: ≥ 0, health and alive never above their start |

**Data:** train only. Marines: exp06. Mixed: exp08 + exp11 (with kite).

## Gate B0 (before any battle)

The MLP must rank intents **better than the frozen model** on the counterfactual sets: regret below the frozen
model's at 2 s and 4 s, paired 95% interval below 0, in marines and mixed. That is M4's first part (`10`).

Reported alongside:
- val errors at 1, 2, 4 s, against frozen and the μ, σ model;
- MLP regret against the μ, σ model's.

**If B0 fails,** the MLP is not a better planner model, and the comparison can't separate the explanations. It
stops, and that is reported.

## Closed loop

- **Planner:** the same as `34` (re-plan every 1 s, commit 2 s, horizon 3 s). Candidates: 7 in mixed (with kite),
  5 in marines.
- **Seeds:** `34`'s seeds (21 x00 000 +, 400 per scenario), so every comparison is paired with the battles already
  played there: fixed kite, μ, σ planner, frozen planner. Nothing is fitted or tuned on those battles, so reusing
  them is valid.
- **Only new policy:** "mlp:h3".

## Rules (fixed once approved)

| | comparison | read as |
|---|---|---|
| **D1 (main: S2, S4, S5)** | MLP planner − best fixed kite (as `34`) | beats / ties / loses, paired 95% bootstrap |
| D2 (all scenarios) | MLP planner − μ, σ planner | the same |
| D3 (S1, S3, flagged) | against best fixed kite and Phase A's best fixed | as `34` C3 |

**Interpretation, fixed now:**
- **D1 beats in at least one main scenario and loses in none:** *the interpretable model limits planning.* A
  flexible model with the same information beats the strongest rule; SINDy does not.
- **D1 loses in every main scenario:** *the setup limits planning.* Even a black-box world model, better at 3-4 s,
  doesn't beat fixed kite with this planner. That points to the near-ties, the short horizon and the dominant
  per-unit rule.
- **Anything in between:** reported per scenario, without a single statement.

## Diagnostics

The same as `35`: intent use, switches per battle, near-tie share, seconds per battle.

## Run time (estimated)

| step | time |
|---|---|
| training, two models | 10-30 min on CPU (timed on 2 epochs first) |
| B0 | a few minutes |
| closed loop (5 scenarios × 400 battles, one new policy) | about 5-10 min |

## Expectations (written before any training)

- B0 passes: the MLP was much better at 4 s before.
- **D1: loses in S2 and S5, about 35% chance of beating kite in S4.** My best guess is "the setup limits
  planning": the oracle with perfect copies of the game but an 8 s horizon is the only planner known to beat kite.

## Open questions for Knut

| | question | draft answer |
|---|---|---|
| Q1 | Give the MLP μ, σ as inputs (same information as the μ, σ planner)? | yes: otherwise a loss could be blamed on missing information |
| Q2 | Reuse `34`'s seeds? | yes: paired with every existing policy; nothing is tuned on them |
| Q3 | Also run an MLP planner with a longer horizon (8 s, like the oracle)? | no, not in this test: it would mix two changes. A later test if D1 loses |
