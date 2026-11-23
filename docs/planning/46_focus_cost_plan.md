# The focus-cost correction on the μ, σ planner, team level, fresh seeds (plan, 23 Nov 2026)

**Status: run on Knut's standing instruction of 23 Nov ("if worth continuing, run the next best experiment").
Rules fixed before any battle.**

## Why

`45`: a perfect model picks a nearest intent in 66% of team-level decisions, while the models over-prefer focus. That
is the bias `03` / `04` found on the counterfactual sets: focus has a short-term cost (fewer Blue units firing at
their own target) that the equations lack. The focus-cost variant (`04`) scales Blue's reach by (1 − c·focus) / f̄,
with c fitted on train (marines 0.096, mixed 0.044). It is applied here, unchanged, to the μ, σ planner.

`45`'s states come from `34`'s seeds, so the variant is chosen after looking at those seeds. **It is therefore tested
on fresh seeds.**

## Policies and seeds

| policy | what |
|---|---|
| `musigma5-focuscost:h3` | the μ, σ planner, five team-level candidates, focus cost (`04`'s c, f̄ per model) |
| `musigma5:h3` | the same without focus cost |
| Phase A's best fixed team intent | S1, S3, S4 hold/focus; S2, S5 advance/focus |
| `oracle5-h3` | the matched ceiling |

**Seeds:** fresh, 23 x00 000 + (S1 23 000 000, …, S5 23 400 000), 400 per scenario, all five scenarios, paired.

## Rules (fixed now)

| | comparison | read as |
|---|---|---|
| **F1** | focus cost − μ, σ planner without it | beats / ties / loses, paired 95% bootstrap |
| F2 | focus cost − best fixed team intent | the same |
| F3 | share of the matched ceiling: (focus cost − best fixed) / (oracle5-h3 − best fixed) | reported |

**Statement:** "the focus-cost correction improves team-level planning" holds if F1 beats in at least one of S1, S2,
S5 and loses in none of them. S3 and S4 are flagged.

## Expectations

- **F1:** beats in S2, ties elsewhere.
- **F2:** ties or a small gain in S1, S2, S5.
- **Run time:** about 15-20 min.
