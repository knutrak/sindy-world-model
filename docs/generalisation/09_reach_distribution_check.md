# Does the distribution of nearest-enemy distances carry reach? (feasibility check, 22 Nov 2026)

**Exploratory, not pre-registered; the criterion below was written before anything was computed.**

## Question

The reach closure guesses how many units are in reach from centroid distance, depth and width. The idea
(Knut, 22 Nov) is to use as states the **mean μ and spread σ of each unit's distance margin to its nearest
enemy**. If those two numbers carry reach, modelling their dynamics is worth doing. If they don't, the idea is
dropped cheaply.

## Definitions

For each alive unit i, its reach margin is

    m_i = min over enemies j of (‖x_i − x_j‖ − r_ij)

where r_ij is the pair reach, capped at 6 as in the typed states. A unit is in reach exactly when m_i ≤ 0.

Per side and unit type, μ and σ are the mean and standard deviation of m_i over alive units. The predicted share
in reach is:

    raw      s = Φ(−μ / σ)
    fitted   s = Φ((c − μ) / (k·σ))        c, k fitted on train (1,000 battles), per scenario

A single alive unit has σ = 0, so the formula becomes a step: in reach exactly when m ≤ c.

## Comparison and criterion

- **Data:** one step at a time on **val** (exp06 val for marines, exp08 val for mixed), at every step where both
  sides are alive, using measured states.
- **Error:** units in reach, |predicted − measured| × 10, summed over types and averaged per side and step.
- **Against:** the frozen pipeline's closure on the same measured states.
- **Passes** if the **fitted** μ, σ version has at least **25% lower** reach error than the closure in **both**
  marines and mixed.
- **Reported alongside:** the raw version, and how often μ and σ stay unchanged from one step to the next (a hint at
  how hard their dynamics will be).

## Result


**Run 22 Nov.** Reach error in units per side and step (val, measured states):

```
scenario  steps  closure  mu-sigma raw  mu-sigma fitted  fitted / closure      c      k  mu unchanged  sd unchanged  count check (max abs)
 marines  83110   0.4840        0.3542           0.3151             0.651 0.0473 1.2019        0.2125        0.2283                    0.0
   mixed 127595   0.5541        0.3408           0.2759             0.498 0.0721 1.2103        0.2561        0.3220                    0.0
```

**Criterion (fitted ≤ 0.75 × closure in both scenarios): passes.**
'count check' = the largest difference between units counted with margin ≤ 0 and the typed reach column (should be 0).

Facts beside the criterion:
- The fitted mapping is close to the raw one (c = 0.05-0.07 map units, k = 1.2), so the mapping from μ, σ to reach
  has almost no free numbers.
- μ and σ change in 74-79% of steps (they are distances, nearly continuous), where the reach count itself is
  unchanged in about 72% of steps (`final_exploration`): smoother targets for SINDy than reach.
- This is one step at a time with **measured** μ and σ: the information is there. Whether μ and σ can be predicted
  forward (their dynamics) is the next, harder question.
