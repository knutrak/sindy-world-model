# Reach from the distribution of distance margins: μ, σ as states (plan, 22 Nov 2026)

**Status: approved by Knut 22 Nov 2026, as drafted; rules fixed before any fit.**
Existing data only (Knut, 22 Nov): if kite terms are dropped, that is reported, and no data is added afterwards.

## Why

The feasibility check (`09`) found that reach, computed from the **measured** mean μ and spread σ of each unit's
distance margin to its nearest enemy, has 35% (marines) and 50% (mixed) lower error than the closure. The mapping
has almost no free numbers. So the information is in μ and σ.

**The question:** can SINDy discover how μ and σ evolve, from a generic library of the state and both sides'
intents? If so, do rollouts and intent ranking get better than with the frozen closure?

## The states

For each side s and unit type a, over alive units i of that type:

    m_i  = min over enemies j of (‖x_i − x_j‖ − r_ij)       reach margin (r_ij = pair reach, capped at 6)
    μ_sa = mean of m_i,    σ_sa = standard deviation of m_i
    p_sa = Φ((c − μ_sa) / (k·σ_sa))                         share in reach; c, k fitted on train as in `09`
    R_sa = A_sa · p_sa                                        units in reach, used by the frozen health equations

Marines have 2 (μ, σ) pairs, mixed 4. Health, alive, distance and target choice stay frozen. Only reach changes:
from closure + decaying offset to the integrated μ, σ.

## The library (generic, the same for every μ′ and σ′ equation)

**Building blocks** for equation (s, a):

| block | what |
|---|---|
| μ, σ, p | this type's margin mean, spread, and the share in reach they imply |
| μ_o, σ_o, p_o | the opposing side, averaged over its types weighted by units alive |
| A_own, A_opp | units alive: own type, opposing side in total |

**State candidates:**
- **Polynomials** of degree ≤ 2 in the 8 blocks: 45 groups.
- **Deaths:** (ln A_own)′ and (ln A_opp)′, each × {1, μ, σ}: 6 groups. A unit dying changes the mean and spread by
  selection; (ln A)′ is the step change at deaths.

That is **51 state groups**.

**Intents:** every state group × {1, adv, hold, wd, focus, kite}. Marines have no kite data, so kite is left out
there.

p enters as a building block because "units walk until in reach" is about the share not yet in reach. It is computed
from the state by the fixed mapping above, not from the closure, and nothing about its role is imposed: STLSQ
decides whether it is used.

**Degree 2 only.** In `08` degree 3 added groups but no accuracy.

## Sparsity and fitting

As in `07` / `08`:
- **Two-level STLSQ,** scaled by root mean square. A state group stays while its norm is ≥ 0.05; an intent
  interaction inside a kept group is dropped below 0.025. Sensitivity runs at 0.025 and 0.1.
- **M-weak (main):** weak form with windows of radius 1 s and 2 s, stride 8 steps.
- **M-step (reference):** one-step SG-9 rates.
- **Bootstrap:** 30 refits by battle (weighted Grams).

## Data (all exists; fit on train only)

| scenario | fit | rollouts | intent ranking |
|---|---|---|---|
| marines | exp06 train | exp06 val | exp06 counterfactual sets (5 candidates) |
| mixed | exp08 train + exp11 train | exp08 val, exp11 val | exp08 cf + exp11 cf_kite_* (7 candidates, as `33` K3) |

No held-out set is read, and nothing is collected.

## Rollouts

μ and σ are integrated with the model's own predictions, from measured start values. σ is clipped at ≥ 0, and reach
is A·p with p as above. The frozen-equality test from `08` still holds: with no μ, σ equations the hook reproduces
the frozen model exactly.

**Blow-up:** NaN, or σ < −0.05 before clipping, or |μ| > 20 map units while units are alive.

## Decision rules (fixed once approved)

| rule | passes if |
|---|---|
| M1 sparse and stable | per equation, ≤ 8 state groups at threshold 0.05 at both radii; kept groups at the two radii overlap with Jaccard ≥ 0.75; every kept group kept in ≥ 90% of 30 bootstrap refits |
| M2 rollouts | M-weak beats frozen on team reach error **and** margin error at 2 s and 4 s (paired 95% interval by battle below 0) in **both** marines and mixed; team health error at 1 s not worse by more than 5%; blow-ups < 1% of start points |
| M3 kite (mixed) | a kite interaction is kept in at least one μ′ or σ′ equation, and the Blue reach bias in 2 s kite windows (exp11 val, as `08` R4) at least halves against frozen |
| M4 intent ranking | on the counterfactual sets, regret at 2 s and 4 s below the frozen model's (paired interval below 0) in both scenarios; in mixed also below always kite/focus at 2 s or 4 s (the rule K3 failed in `33`) |

M2-M4 are evaluated for M-weak at threshold 0.05 at **both** radii, and pass only if they pass at both (as `07`
amendment 8).

**What follows:**
- **M1 and M2 pass:** μ, σ becomes the candidate for **pipeline v2**. Its claim needs a new held-out test (new
  collection, its own pre-registration).
- **M4 passes:** a closed-loop planner test with kite among the candidates, planned separately (S2, S4, S5, fresh
  seeds).
- **M1 or M2 fails:** reported as the answer for μ, σ dynamics; the closure stays.

## Read-outs

- The μ′ and σ′ equations, with their kept intents.
- Term stability across thresholds, window radii and bootstrap.
- Val errors at 1, 2, 4 s (reach, health, margin), margin R² at 4 s, and blow-ups.
- Kite windows: reach bias and errors.
- Regret per ranker, and how often kite is picked.

## Run time (estimated)

| step | time |
|---|---|
| μ, σ state tables (train, val, counterfactual sets) | about 5 min (the `09` code; 11 s for 3,000 battles) |
| fits with bootstrap | about 10 min |
| rollouts and ranking | a few minutes |
| code | 1-2 days (the `08` machinery: library, weak rows, Gram fit, hook, harness) |

## Expectations (written before any fit)

- **M1:** about 50%. The blocks are distances with simple kinematics, so something like μ′ ≈ −v·(1 − p)·adv
  should show up, but correlated polynomials hurt in `08`.
- **M2:** about 50%. Rollouts can drift once μ is predicted rather than measured.
- **M3:** about 50%. Kite acts directly on μ, unlike on health.
- **M4:** about 30%.

## Open questions for Knut

| | question | draft answer |
|---|---|---|
| Q1 | p as a building block: acceptable, or too much structure? | yes: it is a function of the state, fixed before fitting, and STLSQ may ignore it |
| Q2 | The opposing side's μ, σ in the library? | yes: Red closing in moves Blue's margins too |
| Q3 | Degree 2 only? | yes, given `08` |

## Amendments (22 Nov, before any table was built or fit made)

1. **Two libraries** (Knut, after approval): **L-p** as drafted (blocks μ, σ, p, μ_o, σ_o, p_o, A_own, A_opp) and
   **L-nop** without p and p_o (6 blocks: 28 polynomial groups + 6 deaths groups = 34). Rules M1-M4 apply to each;
   the result names the library. L-nop passing is the stronger result (the walk-until-in-reach behaviour found,
   not offered). In both, reach in rollouts is A·Φ((c − μ)/(k·σ)): that mapping is the output, not a candidate.
