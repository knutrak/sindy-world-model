# Findings

The project's claims, with their evidence and how far they can be trusted, organised around the
research questions in the project description (*Data-Driven Dynamic Modeling via Sparse System
Identification*, NTNU / FFI, autumn 2026). Update it after every experiment: add a finding, change
a status, or move a claim to "Superseded". Nothing gets cited in the thesis without an entry here.

**The thesis in one line:** simulator data → a compact, interpretable SINDy world model
$\dot x = f(x, u; \pi)$ → better decisions by an LLM-driven Blue agent (RQ5, the central question).
Phase 1 (identify and validate the world model) ends with a milestone at the end of 2026;
Phase 2 (planning and the LLM) runs January – June 2027.

**Status:** **solid** = clear effect, checked more than one way or on a large sample;
**tentative** = one dataset, one split, a small sample, or an explanation that hasn't been tested
directly; **hypothesis** = a proposed explanation with no test yet.

Last updated 2026-10-29.

---

## Where the project stands (conclusion first)

**What the evidence supports.** At team level, how fast a side loses health follows **how many
enemy units are in range and shooting** far better than the enemy's health. For Blue's losses this
is clear-cut on exp05: Red's units in range explain R² 0.52, Red's health nothing (−0.02), and the
slope, 0.20 per unit in range, is about 93% of a marine's maximum damage rate (F19, F20). For Red's
losses the picture is less clean: Blue's alive units explain the most (0.58), but Blue's health
also explains a fair amount (0.46), and why the sides differ is untested (F22). The law was found
by choosing the right variable and fitting one coefficient, not by SINDy.

**What SINDy has shown so far.** On health alone, SINDy models predict 1–4 s ahead clearly better
than continuing the last second's trend (F14, F23), and the gated form keeps them physically valid
(F13, F15). But their extra terms are mostly **stand-ins for engagement**, which health can't see:
they change sign between datasets (F24) and track the "health as a clock" effect (F7, F16). So
**SINDy on health has not yet found an interpretable law beyond Lanchester.** Whether it can,
once given engagement or alive units as candidate terms, is open (Q3).

**What is not established.**
- **No self-contained world model yet.** The best predictions use the *real* future engagement
  (F23); nothing predicts engagement (Q2).
- **No uncertainty on most numbers.** Almost every result comes from one dataset and one
  train/test split, with no confidence intervals. Differences of a few thousandths in horizon
  error between models are not shown to be real.
- **No generalization tests.** One map, marines only, 6–19 units per side; the held-out set is
  unused (correctly, so far).
- **Several explanations are untested** (F22, F16), and exp05's coverage has only been checked on
  a 2% test run (F18).

**Measured against the project description, the biggest gap is not accuracy but control.** The
world model a planner needs is $\dot x = f(x, u)$: it must say what happens *if Blue does X*. No
model so far has a control input $u$; exp05 keeps Blue's setting fixed for a whole battle; and the
approach phase, where movement decisions matter most, is trimmed away together with distance $d$,
which the project description puts in the minimal state $x_A = [H_B, H_R, d]$. See "Against the
project description" below.

**Where the effort should go (two months to the Phase 1 milestone).** In order: (1) a model with a
control input and distance, fitted on data where Blue's intent changes during battles (Q1, Q2);
(2) intervals on the horizon comparisons (Q4); (3) a second state representation for RQ2 (Q5);
(4) a clean world-model interface and a first test with a simple planner (Q6). New scenarios for
RQ4 come after that.

---

## Against the project description

### Research questions: where each stands

| RQ | Question | Status | Evidence | Gap |
|---|---|---|---|---|
| RQ1 | Can SINDy identify a compact, interpretable world model from game data? | **Partly.** Compact, physically valid models of health after first contact exist; the interpretable law came from choosing variables, not from SINDy's search | F13–F17, F19–F21, F24 | No control input, so not yet a *world* model; correction terms not stable across datasets |
| RQ2 | How does the state representation affect accuracy, complexity, interpretability? | **Started.** Only Level A variants: health only, and health with alive / engaged as drivers | F6, F7, F19, F23 | Distance not in the state; no Level B (groups) or C (per unit) comparison |
| RQ3 | How accurately does it predict over different horizons? | **Measured, without intervals.** Error vs. horizon 0.5–4 s against persistence and linear extrapolation, on unseen battles | F14, F23 | One split per dataset; no intervals; no horizon longer than 4 s; actions not applied during prediction |
| RQ4 | Does a model from some scenarios work in others (force size, unit types, properties)? | **Barely started.** exp05 varies force size (6–19) and starting state within one model | F18, F19 | One map, one unit type, no unit properties varied, held-out set unused |
| RQ5 | Does a world model improve an LLM agent's decisions? | **Not started** (Phase 2) | – | Needs RQ1 with control, and a world-model interface |

### Phase 1 work plan (section 5 of the description)

| Step | Status |
|---|---|
| 1 Understand the environment | done (F1–F3, README "About SMAClite") |
| 2 Collect and analyse data | done: exp01–exp05, coverage checks |
| 3 First world model (Level A, SINDy) | done, on health without distance (F12–F17) |
| 4 Refine: states, libraries, methods | in progress: gated library, firepower drivers; no discrete-time model, no Level B |
| 5 Validate on unseen data vs. baselines over horizons | in progress: horizon metric exists (F14, F23); intervals missing |
| 6 Generalise across scenarios | started: force size within exp05 only |
| 7 Package behind a clean world-model interface | not started |

### Where the description changes earlier reasoning

- **Distance and the approach phase are back in scope.** F5 dropped distance as a state because,
  after trimming to the first damage, it carries little information *about health*. That holds for
  modelling attrition, but a world model must also predict what happens when Blue advances,
  holds or withdraws, which acts through distance and decides engagement. Engagement (F19, F23) is
  then something to predict from $d$ and $u$, not an input.
- **The control input should be an intent, and our policy parameters already are one.** The
  description suggests an abstraction layer: $u$ = aggressiveness, movement direction, target
  prioritisation, translated into unit commands by a simple controller. `RetreatPolicy`'s
  parameters (target rule, aggression, retreat thresholds) are close to that; what's missing is an
  explicit movement intent (advance / hold / withdraw).
- **Intent must change during battles.** exp05's one setting per battle was the right choice for
  fitting $f(x)$ without control, but a planner (MPPI) evaluates *sequences* of actions. Identifying
  $f(x, u)$ needs data where $u$ switches within battles (`SwitchingPolicy` with
  `segment_seconds`), and where it is logged per step (it already is).
- **Discrete events are a named research topic.** The description expects difficulty with "a unit
  being destroyed or a weapon entering a cooldown". F6, F8 and F19 are findings on exactly that:
  health is continuous enough, alive is a step function, and firepower follows units, not health.
- **Unit properties are a route to RQ4.** The firepower coefficient matched a marine's damage per
  second within 7% (F20). If coefficients scale with unit properties ($k \approx$ efficiency ×
  damage / cooldown / hp), scenario parameters $\pi$ enter the model physically: a concrete test
  for other unit types.
- **The description asks to compare continuous and discrete time** ($x_{k+1} = F(x_k, u_k)$). A
  discrete-time model at the planner's step also avoids differentiating the health staircase (F8).

### Proposed refinements to discuss with the supervisors

The description asks for changes to RQ1–RQ4 to be agreed and documented. Candidates:

- **RQ1:** add "...and does it recover known attrition mechanisms (firepower of units in range)
  when given suitable variables?", since variable choice decided interpretability (F19–F21).
- **RQ2:** frame the comparison as health-only vs. health + units + distance (Level A variants),
  with Level B as a stretch: with marines only, natural groups are "in range / out of range" rather
  than roles.
- **RQ3:** include the error *with the planner's actions applied* along the horizon, which is how
  the planner will use the model.

---

## Simulator and data

**F1 · Blue's win rate rises with aggression and falls with step_mul.** *Solid (effect); hypothesis (cause).*
exp01, 500 battles per setting: at aggression 0.9, 86% / 65% / 31% / 4% for step_mul 1 / 2 / 4 / 8
(README table). The proposed cause, that Red reacts every tick while Blue decides every step_mul
ticks and its random actions last a whole step, is plausible but has not been tested.

**F2 · Battle length in seconds doesn't depend on step_mul.** *Solid.*
About 7.5–9.5 s at every step_mul in exp01 and exp02 (recap notebook, Part 4). step_mul changes how
finely a battle is recorded and how often Blue decides, not the battle itself.

**F3 · A fixed spawn with aggression 1.0 replays one game; randomized spawns give real variety.** *Solid.*
exp01 at aggression 1.0: 1–2 distinct outcomes per 500 battles. exp02: 74–98 distinct outcomes per
100 battles at every setting (README).

**F4 · Randomized spawns lower Blue's win rate at step_mul 1–2, and distance alone doesn't explain it.** *Tentative.*
exp02 battles that spawn 13.5–15.5 apart (close to exp01's 14.2) win 45% at `a090_s1`, against
86% in exp01 (n = 22). The bearing analysis that hinted at an angle effect was done on an earlier
exp02 collection, with n = 9, and hasn't been redone. Not isolated; a bearing-only dataset would.

**F5 · After trimming, spawn distance carries almost no information about health.** *Solid for exp02; unchecked for exp05; conclusion revised.*
exp02 `a090_s1`: spawn distance std 2.10, distance at the first damage std 0.43; spawn distance
mostly decides *when* the fight starts (correlation 0.98 with the trim point). So distance was
dropped as a state. exp05 includes starts in contact, where this may no longer hold. **Revised
2026-10-29:** this is about modelling attrition after contact. For a world model with movement
intents, distance (and the approach phase) is needed; see "Against the project description".

## States

**F6 · Alive units are not rounded health: damage is spread.** *Solid (exp03, one dataset).*
`10m_attack_a090_s1_v1`: alive equals ⌈10 × health⌉ only 23–33% of the time; on average 1.4–1.7
more units are alive than focused damage would leave; each side shoots at about 2.5 targets.

**F7 · On equal, full-health starts, health works as a clock for engagement.** *Solid.*
exp03: the share of a side's alive units in range rises from 20% at full health to 92% below 20%
health; correlation −0.72. exp05's design reduces it to −0.19 (2% test run; see F18).

**F8 · The health derivative is a well-defined average rate at half-second scale.** *Solid.*
exp03: 13% of 0.25 s windows contain no damage, 1% of 0.5 s windows, none of 1 s windows; the mean
rate is stable at about −0.10/s. The fitted equations barely change across smoothing windows 5–17
and downsampling 1–4 (notebook 04 Part 4).

**F9 · Stacking episodes is correct if each is differentiated first; episode edges are noisy.** *Solid (method).*
Shuffling all 31,409 training rows changed the coefficients by 10⁻¹⁶; differentiating *after*
stacking gave meaningless coefficients (e.g. −7.78 BR). Dropping the 4 edge points per episode (10%
of rows) barely moved Blue's equation but changed Red's noticeably (the coefficient of Red health
squared went from 0.283 to 0.168): Red's equation is poorly determined (see F15).

## Evaluation

**F10 · Predicting the winner from midway is capped by the information in the state, not by the model.** *Solid.*
exp03 `a090_s1`, all 500 battles: leader baseline 0.862; the best classifier using health 0.858,
+ alive 0.862, + engaged 0.874. Winner accuracy is a poor metric for these models. Also, 100 test
battles give about ±3.5 points of noise: the same baseline scored 0.93 on the test split.

**F11 · Evaluating at the real end time leaks information.** *Solid (method).*
The first winner metric stopped the simulation at the real end of the battle, which a model
wouldn't know, and counted models as right that never eliminate anyone. Replaced by
`simulate_to_elimination` and, for world-model use, `evaluate_horizons`.

## Models on exp03 (`10m_attack_a090_s1_v1`, one split, 400 / 100 battles)

**F12 · Lanchester on health gives 73–78% of full marine firepower.** *Solid (number); superseded (interpretation).*
a = 0.160, b = 0.170 against 0.22/s for a full team firing. It was first read as "efficiency lost
to moving and overkill"; F20 shows it is mostly health mixing in units that aren't shooting yet.

**F13 · Polynomial SINDy on health predicts healing.** *Solid.*
E.g. `(health_red)' = −0.356 B + 0.197 R`: once Blue is gone, Red's health grows. healing_share
8–27% for the polynomial fits; 0 for Lanchester and gated (notebook 04 Part 6).

**F14 · Health-only SINDy beats naive extrapolation a few seconds ahead.** *Tentative.*
Error 4 s ahead: degree-2 polynomial (5 terms) 0.036, gated degree 1 (6 terms) 0.039, Lanchester
0.091, last slope 0.092, persistence 0.393. One dataset, one split, no intervals.

**F15 · Gated degree 1 is stable; degree 2 isn't.** *Tentative.*
10 bootstrap resamples of training battles: degree 1 terms spread at most 0.017, no sign flips;
degree 2 terms spread 0.06–0.15 with sign flips. Bootstrap by battle, one dataset.

**F16 · The gated model's extra terms stand in for engagement ramp-up.** *Hypothesis, with support.*
Firepower per unit of health rises during the battle (0.08 → 0.34 along a typical battle), as
the in-range share does (F7). Supported by F24 (the terms change between datasets), but not tested
directly: the test is whether they vanish when engagement is a candidate term (Q3).

**F17 · The dense degree-2 model was dense only because of a too-low threshold.** *Solid.*
Threshold 0.005 → 12 terms; 0.05 → 5 terms with the same 4 s error (0.036).

## exp05 (sampled conditions, one policy setting per battle)

**F18 · The exp05 design breaks the diagonal and weakens the health clock.** *Tentative (2% test run).*
On 400 battles: corr(Blue health, Red health) 0.28 (exp03: 0.97); corr(Red health per unit,
share in range) −0.19 (exp03: −0.72); Blue win rate rises smoothly with force ratio (6% → 95%).
The coverage summary has not been reviewed on the full 20,000.

**F19 · On exp05, units explain a side's losses better than health does, clearly so for Blue's losses.** *Solid (5,000 battles).*
One-variable fits through the origin, 4,500 training battles (walkthrough notebook, section 4):

| Losses of | opponent's health | opponent's alive | opponent's engaged |
|---|---|---|---|
| Blue | −0.02 | 0.15 | **0.52** |
| Red | 0.46 | **0.58** | 0.42 |

For Blue's losses, Red's health is useless and Red's units in range are the driver. For Red's
losses, Blue's health does explain a fair amount, though less than Blue's alive units. Why the two
sides differ isn't settled (F22). A side's *own* health explains little either way (Blue 0.09,
Red −0.25).

**F20 · Damage per shooting marine is about 93% of the game's maximum.** *Solid for Blue's losses; tentative for Red's.*
Blue's losses per engaged Red unit: 0.202/s = 9.1 hp/s, against 9.84 hp/s for a marine (93%).
Red's losses per engaged Blue unit: 0.208/s (95%), but on a driver that overstates firing (F22),
so that number is less reliable. Not yet checked per policy setting (Q8).

**F21 · The "own health" term in Blue's equation is a threshold artefact plus a stand-in.** *Solid.*
Without a threshold, Blue's equation is −0.026 − 0.071 B − 0.040 R; threshold 0.07 drops the Red
term. Blue's health correlates more with Red's engaged units (0.49) than Red's health does (0.33).

**F22 · Blue's engaged count overstates how many Blue units fire.** *Hypothesis.*
Proposed cause of the asymmetry in F19: Blue takes random actions (1 − aggression) or retreats
while a target is in range; Red's AI always fires. Test: count attacking Blue units from
`actions/` and check it beats alive (Q8).

**F23 · Knowing engagement roughly halves the error a few seconds ahead.** *Tentative.*
Error 4 s ahead, 500 test battles: firepower model with the *real* future engagement 0.032;
degree-2 SINDy 0.052; gated degree 1 0.061; last slope 0.078; Lanchester 0.085 (now worse than
last slope). The firepower row is an upper bound (it reads the future), not a usable model.
Degree 2's stability on exp05 hasn't been checked.

**F24 · The gated model's correction terms are not stable across datasets.** *Tentative.*
Blue's equation, gated degree 1: exp03 `−0.408 R + 0.150 RB + 0.176 R²`; exp05
`−0.235 R − 0.100 RB + 0.196 R²`. The RB term changes sign. Consistent with F16: the corrections
reflect the data's mix of conditions, not a general law. (exp03 is on a per-side scale, exp05 on
the common scale; they coincide for a full-health 10v10 start, so the sign change isn't a scale
effect.)

## Policies

**F25 · The target rule changes how concentrated damage is.** *Tentative (60 battles each).*
Extra Red units alive beyond perfectly focused damage: 2.3 (nearest) vs 0.7 (weakest, focus).
Win rates 63% / 80% / 68% (±12 points at this n). Focus and weakest behave almost the same here.

**F26 · Retreating weak units loses battles and doubles their length.** *Tentative (60 battles each).*
Win rate 15% vs 63%; length 22.5 s vs 10.3 s. In exp05 this made retreat battles 24% of battles but
46% of steps, so the retreat shares were lowered (now 14% of battles, 26% of steps).

---

## Superseded claims

- **"Degree-2 SINDy beats the leader baseline from midway (0.84 vs 0.73)"** (README, about a 5v5
  dataset): measured with the old metric that leaks the real end time (F11), on a small test split.
  Winner-from-midway is capped by the state anyway (F10). Don't cite.
- **"Lanchester's 73–78% is efficiency lost to moving and overkill"**: mostly an artefact of using
  health as the driver (F12, F20).
- **README findings for exp01/exp02 before 2026-10-27** came from earlier collections no longer on
  disk; the README has since been recomputed on the data on disk.

## Open questions, in order of importance

1. **Q1 · Control (RQ1, RQ5).** Can SINDy identify $\dot x = f(x, u)$ with $u$ an intent (target
   rule, aggression, movement: advance / hold / withdraw)? Needs a movement intent in the policy and
   a dataset where intent switches within battles, without trimming the approach phase.
2. **Q2 · Distance and engagement (RQ1, RQ2).** With $d$ in the state, does SINDy predict engagement
   (units in range) from $d$ and $u$, and does the firepower law (F19, F20) follow? This is what
   makes the model self-contained (F23).
3. **Q3 · Does SINDy rediscover the firepower law?** Given units in range (or alive) as library
   terms, does it pick ≈ −0.2 × units in range on its own, and what does it add (RQ1)?
4. **Q4 · Are the model differences real (RQ3)?** Repeat the horizon comparisons over several splits
   or bootstrap them; report intervals; extend horizons beyond 4 s; continuous vs. discrete time.
5. **Q5 · A second state representation (RQ2).** Level A with health, units and distance vs. health
   only, and if time allows a group-level state.
6. **Q6 · A world-model interface and a first planner (Phase 1 step 7, Phase 2 step 1).** Wrap the
   model as `predict(state, intents, horizon)` and test a simple sampling planner without an LLM: does
   planning with it beat the fixed policies?
7. **Q7 · Generalisation (RQ4).** Other unit types, unit properties and larger forces; do the
   coefficients scale with damage, cooldown and hp? Plus the held-out set for final numbers.
8. **Q8 · Policy effects and the Blue/Red asymmetry.** $k$ per target rule and aggression; Blue's
   attack share from `actions/` (F22). Feeds Q1.

---

## Template for an experiment note

Copy into the top of a notebook (before running) and its last cell (after).

```
Question:       what do I want to know? (one sentence)
Prediction:     what do I expect, and why?
Would change my mind if:
Data:           dataset, number of battles, split

--- after running ---
Answer:         one or two sentences
Evidence:       key numbers, with where they come from
How sure:       solid / tentative / hypothesis, and why
Findings log:   which F-numbers are added or changed
Next question:
```
