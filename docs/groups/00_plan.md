# Front / rear group states (plan, 25 Nov 2026)

Asked for by Knut on 25 Nov. Written **before** any fit. **Exploratory:** train and val only; val has been used for many
choices before. No held-out set is read.

## Question

Does splitting each side by position (front half, rear half) improve the model, beyond what splitting it into two
arbitrary halves does?

## Definition

- At the start of a window, the alive units of a side are projected on the centroid-to-centroid axis. The half
  nearest the enemy (rounded up) is the **front**, the rest the **rear**.
- Labels are **fixed within a window** (no relabelling), because relabelling would move health between groups.
- Training windows: 4 s, non-overlapping. Val windows: one per start point (every 1 s, as the harness), up to 4 s.
- Model: the pipeline with two groups as if they were two unit types with marine statistics. Reach, health, alive
  and the position offset are per group; distance and formation are per team, as before.

## Models, all fitted on the same training windows

| | what |
|---|---|
| T | one group per side (the frozen pipeline's recipe) on the same windows |
| FR-c | front / rear, fitted target choice (it can send more fire to the front) |
| FR-r | front / rear, proportional target choice |
| P-c | **placebo:** two random halves, fitted target choice (same number of parameters as FR-c) |
| frozen | the frozen marines model, scored on the same val windows |

## Read-outs (marines, exp06 val; health in ten marines, mean per side)

- **G1 gain:** FR-c health error below T at 2 s and at 4 s (paired 95% interval, battle bootstrap, below 0).
- **G2 not just parameters:** FR-c below P-c at 2 s and at 4 s (same rule).
- **G3 structure:** each of the four group equations keeps its two firepower terms and nothing else; no wrong signs; all
  eight coefficients 0.75-1.0 of the game's 0.213.
- Also reported: margin R², alive error, one-step R², the share of damage taken by the front.
- **Replication:** the same on marauders (exp09), reported, not judged.

**"Useful"** only if G1 and G2 both hold. G1 without G2 means that two arbitrary halves do as well.

## Expectations (Claude's, before any fit; Knut has not written his own)

- About 35% that G1 holds, 25% that G1 and G2 both hold: one-step, front-half health added only 0.010-0.012 R².
- About 70% that G3 holds: the halves have identical units, so k should stay near 0.18.
- The front takes clearly more damage than the rear in approach and first contact; less clear later.

## Amendment 1 (post hoc, 25 Nov, after the first run)

The plan had no placebo for FR-r. Added: **P-r**, random halves with proportional target choice, and the health error per group
(front, rear) for FR-c and FR-r. Judged on nothing; only reported.
