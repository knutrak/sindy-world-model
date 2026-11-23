# A hybrid (switching) movement law: feasibility check (23 Nov 2026)

**Exploratory; the criteria below were written before anything was computed** (Knut's go-ahead, 23 Nov). It follows
`14`: per-unit movement is a walk/stop switch, which a single regression cannot represent.

## The hybrid law

Each alive unit is in one of two modes at each step:

| mode | law to discover |
|---|---|
| **walk** (next-step speed > 2.9) | speed v (one number per side and type) and heading (towards the unit's own target if it has one, else its nearest enemy) |
| **stop** (next-step speed < 0.3) | speed 0 |

The **switch** (walk or stop) is a sparse classifier on the same unit features × Blue's intent as `14`:
- margin and its gates;
- weapon readiness;
- danger;
- own target in range;
- for hold, whether contact has been made.

Steps with speeds in between (collision avoidance) are counted and reported, not classified.

## Methods

- **Switch:** L1-penalised logistic regression. The penalty is the weakest that keeps **at most 8 terms**; a
  depth-3 decision tree is reported alongside.
- **Walk mode:** median next-step speed, and the cosine between the velocity and the direction to the unit's own
  target (or nearest enemy when it has none).

**Data:** the per-unit rows of `14` (fit on train, evaluate on val), extended with the target heading.

## Criteria (per side and unit type, on val)

| part | passes if |
|---|---|
| **switch** | accuracy ≥ 90% with ≤ 8 terms (walk and stop steps only) |
| **walk speed** | median within 5% of 3.15 |
| **walk heading** | median cosine to own target (or nearest enemy) ≥ 0.95 |

**What follows:**
- All three pass for Red and for Blue's main intents: U2 (geometry alone, against the closure) is planned on the
  hybrid law.
- Otherwise: reported, as a statement of what the switch depends on.

## Result

**Run 23 Nov** (`experiments/generalisation/unit_geometry.py hybrid`; val rows of `14`).

```
scenario side     type  walk share  stop share  in between  switch accuracy (L1, <= 8 terms)  terms  tree depth 3 accuracy  majority class  walk speed (median)  walk heading cos (median)  passes
 marines blue   MARINE       0.270       0.625       0.105                             0.968      8                  0.950           0.698                 3.15                     -0.016   False
 marines  red   MARINE       0.405       0.434       0.161                             0.952      8                  0.950           0.518                 3.15                      0.965    True
   mixed blue MARAUDER       0.260       0.568       0.172                             0.929      8                  0.868           0.686                 3.15                     -0.411   False
   mixed blue   MARINE       0.267       0.566       0.167                               NaN      0                  0.909           0.680                 3.15                     -0.444   False
   mixed  red MARAUDER       0.197       0.523       0.280                             0.883      8                  0.888           0.726                 3.15                      0.955   False
   mixed  red   MARINE       0.260       0.460       0.280                             0.831      7                  0.834           0.639                 3.15                      0.985   False
```

Switch terms (standardised coefficients; positive = more likely to walk):

- marines blue MARINE: +1.40·g0·adv +0.95·g0·hold*contact +0.25·g0.5·adv +0.02·g0.5·hold*contact -0.20·ready·1 +1.83·danger·wd -0.91·tir·1 +0.01·tir·wd
- marines red MARINE: +2.26·m·1 +0.04·g-1·1 +2.32·g0·1 -1.32·g0.5·1 -0.72·ready·1 +0.24·ready*g0·1 +0.63·danger·1 -1.28·tir·1
- mixed blue MARAUDER: +0.33·1·kite +0.07·g-0.5·kite +0.10·g0·kite +0.59·g0.5·adv +0.21·g0.5·hold*contact +0.17·ready·kite +0.95·danger·wd -0.69·tir·1
- mixed blue MARINE: (no L1 model with at most 8 terms was found)
- mixed red MARAUDER: +0.22·m·1 +0.09·m2·1 +0.56·g-0.5·1 +0.31·g0.5·1 -0.24·ready·1 +0.23·ready*g0·1 +0.18·danger·1 -1.59·tir·1
- mixed red MARINE: +0.57·m2·1 +1.03·g-0.5·1 +0.32·g0.5·1 -0.45·ready·1 +0.37·ready*g0·1 +0.31·danger·1 -1.27·tir·1

### Read-outs against the criteria

| part | result |
|---|---|
| walk speed | **passes everywhere**: median exactly 3.15, the game's speed |
| walk heading | **passes for Red** (median cosine 0.955-0.985 to own target or nearest enemy); **fails for Blue** as defined (−0.02 to −0.44), because Blue's training mix includes withdraw and kite, which walk away on purpose |
| switch | **passes for marines** (Blue 96.8%, Red 95.2%, 8 terms) and mixed Blue marauders (92.9%); **fails for mixed Red** (83.1% marines, 88.3% marauders) and for mixed Blue marines (no L1 model with ≤ 8 terms; depth-3 tree 90.9%) |
| overall | **passes only for marines' Red**; the condition for going on to U2 (Red and Blue's main intents) is not met |

Facts beside the criteria:
- Blue's walking heading **per intent** (val, full-speed steps): advance 0.90 / 0.90, hold 0.87 / 0.84, withdraw −0.92 /
  −0.94, kite 0.65 (marines / mixed): towards the enemy when advancing or holding, away when withdrawing. The heading
  law is intent-dependent, which the criterion as written did not allow for.
- The switch is driven by the expected terms: **own target in range** (stop; −0.7 to −1.6) and being **out of
  range** (walk; the gates), plus **danger under withdraw** (walk away; +0.95 to +1.83).
- What stops the mixed scenarios: **28% of Red's steps are in between** (slowed or deflected by collision avoidance
  in denser mixed formations, against 16% in marines), and the switch accuracy there is only 83-88%.

**Reading:** the hybrid structure is right (two modes, speed 3.15, heading by intent, a switch on target-in-range),
and it is accurate for marines. Mixed armies add crowding that neither mode captures. Collision avoidance is the next
thing to model; it is the natural first step for next semester.
