# Findings so far, rated (20 Nov 2026)

A summary of the final exploration (branch `final-exploration`, 18-19 Nov) and the marine + marauder work
(branch `mixed-marauders`, 19-20 Nov). Details: `docs/final_exploration/REPORT.md`, `docs/mixed/PROGRESS.md`.

Two ratings per finding, 1-10, my judgement:

- **Quality**: how solid the evidence is. Pre-registered and held-out scores high; post-hoc, val-only or
  selected-sample results score lower.
- **Usefulness**: how much it helps the thesis (the RQs, the argument, or the honesty of its limitations).

## A. The marine model (A\*-2k)

| # | finding | quality | usefulness |
|---|---|---|---|
| A1 | **The coefficients read as game physics.** Closing speeds 6.0 / 3.2 against the game's 6.3 / 3.15. Firepower 0.177 / 0.186, i.e. 83-87% of the "everyone in reach fires" rate. | 8 | 9 |
| A2 | **The alive law fitted on the rollout's regressor** (A' = p·k·R·A/H) removes errors-in-variables attenuation. Gains in 5/5 cv folds and on test2, and p becomes stable across derivative estimates, which fixed A's G3b failure. | 8 | 8 |
| A3 | **Pre-registered test2** (A\*, B, S; 5/5 expectations met; matches val, so no overfitting to val). A\*-2k itself was not run on it. | 9 | 8 |
| A4 | **One shared k was wrong.** Red's AI acquires targets at 5.375, Blue shoots at 5.75. k_B < k_R is real (non-overlapping intervals). The separate-k model ties A\* and transfers the margin to stalkers better (0.195 vs 0.122). Measuring Red at 5.375 overcorrects. | 7 | 6 |
| A5 | **Gates G1-G8 pass for A\* / A\*-2k** (post-hoc on val, cf, stalkers; G7 n/a for A\*-2k). | 5 | 6 |
| A6 | **Reliable horizon 2-3 s.** The MLP is better everywhere (health 4 s 0.033 vs 0.053): the price of interpretability, in numbers. | 8 | 8 |
| A7 | **Trajectory fitting (B, S) buys margin accuracy at 2-4 s**, but the coefficients drift from physics: closing speed 7.7 against 6.3, offset decays with no game meaning. | 8 | 7 |
| A8 | **Ranking intents:** rank correlation ≈ 0.5 at 4 s, far above random, but no model picks the best intent more often than "always advance/nearest". Exploratory, on `cf`. | 6 | 8 |
| A9 | **Withdraw-heavy battles are where it fails.** Distance error grows faster than real movement. At equal battle length, battles with withdraw still have a higher whole-battle error. | 6 | 7 |

## B. Marines + marauders (exploration, val only)

| # | finding | quality | usefulness |
|---|---|---|---|
| B1 | **SMAClite's fixed `AGENT_TARGET_RANGE = 6`.** Agents can't be ordered to fire beyond 6, whatever their range, and Red's AI only acquires within its scan radius. Marauders between 6 and 7.125 fire 0% / 7% of the time. Verified in the code and the data. | 9 | 8 |
| B2 | **The 2×2 damage matrix matches the game.** With reach capped at 6, all 8 entries are 0.82-0.94 of the game rate, with Blue and Red alike and tight bootstrap ranges. One step, measured target choice. | 7 | 8 |
| B3 | **The structure transfers to marauders.** A refit finds the same 11 terms. With the cap, the scaled marine model transfers (margin R² 2 s 0.36 vs −0.04 uncapped). | 6 | 7 |
| B4 | **The firing share is not a unit property but an environment effect.** 0.54 → ≈ 0.85 for marauders once reach is counted at 6. | 8 | 7 |
| B5 | **The per-type state helps rollouts but not one step ahead.** Health −10% and margin R² 4 s +0.06-0.11 against A\*-2k on team totals. It needs the withdraw rule, and the uncapped fit predicted a little better. | 6 | 6 |
| B6 | **Collinearity caveat.** With firing ≈ 0.85 × reach, which description STLSQ keeps depends on the library; the per-firing-unit rate isn't identifiable with reach as a candidate. | 7 | 6 |
| B7 | **Target choice is readable.** Fire shifts to the type further forward, and Red over-targets marauders. But the fitted closure ≈ the proportional rule in rollouts. | 6 | 5 |
| B8 | **One distance is defensible on average.** Types stay level on average (≤ 0.2); the per-battle spread is about the formation depth. | 7 | 4 |
| B9 | **The per-type alive law is weak.** p is 0.35-0.53 and the one-step R² 0.05-0.16; alive error is no better than the team model's. | 6 | 4 |
| B10 | **Ranking in mixed battles: the same pattern as marines**, and the per-type state doesn't help ranking. | 6 | 5 |
| B11 | **Removing withdraw (imitation):** better whole-battle error and winner accuracy, but the coefficients get less physical. Selected battles, biased towards short ones. | 4 | 6 |

## Overall, for the thesis

| part | quality | usefulness | comment |
|---|---|---|---|
| Marine model A\*-2k as the core (A1-A3, A6) | **8** | **9** | Strong: interpretable, physics-checked, pre-registered test (for A\*), honest horizon and MLP gap. |
| Its limitations (A7-A9) | 7 | 8 | Valuable negative results: trajectory fitting vs physics, ranking vs fixed rule, withdraw. |
| Marine + marauder extension (B) | **6** | **7** | A promising "the method extends" result with a real discovery (B1). Exploratory: val only, choices made after looking, test3 unread. |
| Removing withdraw (B11) | 4 | 6 | An imitation; it only shows the direction of the trade-off. |

**My reading (hedged).** The thesis already has a solid core: an interpretable world model whose
coefficients match the game, a fixed estimation bias, and an honest account of where it stops (2-3 s,
withdraw, ranking no better than a fixed rule). The strongest new piece from the mixed work is the chain
B1 → B2 → B4: an environment rule showed up as "odd coefficients", and the physics check found it. That
argues *for* checking SINDy coefficients against known rules, which is close to your RQ1.

The mixed model's predictive gain (B5) is real on val, but smaller and more conditional than the physics
result. Treat it as an extension, not a second core model, unless it goes through gates and a
pre-registered test3 run.

**Gaps that would raise the scores most:**

1. A held-out number for A\*-2k: a fresh set or an agreed use of an existing one.
2. Pre-registered test3 for the mixed model (B2, B5).
3. A clean no-withdraw collection, if that framing matters (B11).
4. Checking whether the 6-unit cap also explains the stalker transfer loss (likely, untested).
