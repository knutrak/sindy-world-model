# 2026-10-27 · Project recap: what has been built, why, and what it showed

A full recap before the large exp05 collection. It follows the project in the order the thinking
went: the question, the simulator, the data, the states, the models, how to judge them, what
went wrong with the data, and how exp05 is designed to fix it. Every step links to the function
that does it. Numbers come from the notebooks and the checks run along the way; where they come
from one dataset, that dataset is named.

Companion notebooks: [`00_recap`](../notebooks/00_recap.ipynb) runs the whole pipeline once,
[`01_models_explained`](../notebooks/01_models_explained.ipynb) walks through the model code
function by function.

---

## 1. The question

Can sparse system identification (SINDy) learn a **compact, interpretable model of how a battle
evolves**, good enough to serve later as a **world model** for planning: "if the battle is in
this state now, where will it be in a few seconds?"

Two ideas run through everything below:

- **Interpretable means a few terms with physical meaning.** The classical reference is
  Lanchester's square law, $\dot B = -aR$, $\dot R = -bB$: each side's losses are proportional to
  the other side's strength. SINDy is interesting if it finds what real combat adds to that law,
  and if those additions mean something.
- **A world model is judged by how well it predicts a short time ahead,** not by whether it can
  name the winner of a whole battle. This took a while to see (section 6).

---

## 2. The pipeline and its principles

```
SMAClite battle ──envs──▶ Episode: per-unit snapshots, actions, controls
                ──io────▶ dataset folder data/raw/<experiment>/<dataset>/
                ──states▶ team-level table (health, alive, distance, engagement) per step
                ──models▶ trajectories → fit (SINDy / Lanchester / gated) → simulate → score
                ──runs──▶ runs/<date>_<name>/ (config, equations, metrics, figures)
```

Principles, all enforced in code:

- **Log raw snapshots once, derive everything else.** States are never stored, so a new state
  definition never needs new battles. That is why the normalization problem in section 9 can be
  fixed after collecting.
- **One seed describes a whole episode.** The environment, the policy and every sampler are driven
  by the episode's seed (separate streams, see section 8), so any battle can be replayed exactly.
- **Provenance.** Every dataset and run records the git commit, whether there were uncommitted
  changes, and package versions: [`code_info`](../src/sindy_wm/provenance.py#L25).
  [`create_dataset`](../src/sindy_wm/io/storage.py#L78) refuses uncommitted code unless told
  otherwise.
- **Nothing is overwritten.** Datasets and runs get new names instead.
- **Split by episode, fit one trajectory per episode,** and keep a held-out set for final results.

---

## 3. The simulator layer (`sindy_wm.envs`)

[SMAClite](https://github.com/uoe-agents/smaclite) is a light Python version of the StarCraft
Multi-Agent Challenge. Blue is our policy; Red is the game's built-in AI, which attack-moves
towards Blue. Only `envs/` imports SMAClite.

### Creating a battle

- [`make_env`](../src/sindy_wm/envs/scenarios.py#L63) builds an environment from a scenario
  name (`configs/scenarios/5m_vs_5m.json`, `10m_vs_10m.json`). It can override where the teams
  start (`spawn=`) and how many units each side has (`unit_counts=`).
- [`StepMulWrapper`](../src/sindy_wm/envs/scenarios.py#L43): `step_mul` is game ticks per step
  (16 ticks = 1 s). SMAClite stores it as a global, so the wrapper sets it before every step.
  `step_mul` sets both how finely a battle is recorded and how often Blue decides.
- [`sample_opposing_spawns`](../src/sindy_wm/envs/spawn_sampling.py#L23) draws a distance and
  bearing between the two groups, both centred on the map (the terrain is open, so only the
  relative layout matters). [`formation_half_width`](../src/sindy_wm/envs/spawn_sampling.py#L94)
  knows how SMAClite lays groups out (a square grid), so groups never overlap.

### Playing and recording

- [`play_episode`](../src/sindy_wm/envs/episode.py#L37) runs one battle to the end (SMAClite has
  no time limit, so `max_steps` stops it), optionally with damaged starting health
  (`start_health=`, applied right after reset), and returns an
  [`Episode`](../src/sindy_wm/envs/episode.py#L13):
  - `snapshots`: every unit at every step (position, hp, velocity, its target, whether the target
    is in range), recorded by [`SnapshotLogger`](../src/sindy_wm/envs/smaclite_snapshot.py#L30).
    SMAClite deletes dead units; the logger keeps them with `hp = 0`.
  - `actions`: Blue's action per unit per step.
  - `controls`: the policy's setting at every step ([`Policy.setting`](../src/sindy_wm/envs/policies.py#L30)).

### Policies ([`policies.py`](../src/sindy_wm/envs/policies.py))

| Policy | What it does | Why it exists |
|---|---|---|
| [`AttackNearestPolicy`](../src/sindy_wm/envs/policies.py#L61) | shoot the nearest enemy in range, else walk towards the nearest; with probability `1 − aggression` a random action | the first policy; aggression is the source of variety between battles |
| [`TargetChoicePolicy`](../src/sindy_wm/envs/policies.py#L106) | `rule` = nearest (identical to the above), weakest, or focus (whole team on one target) | controls how damage is spread, so how fast enemy units die for the same damage |
| [`RetreatPolicy`](../src/sindy_wm/envs/policies.py#L188) | target rule, plus: weak units, or a losing team, move away from nearby enemies | makes engagement go up *and* down; a real planning decision |
| [`PolicySpace`](../src/sindy_wm/envs/policies.py#L269) + [`SwitchingPolicy`](../src/sindy_wm/envs/policies.py#L303) | draw random `RetreatPolicy` settings, one per battle or a new one every few seconds | varies the policy within one dataset; logged per step for later models with control |

On 60 battles each (10v10): the weakest and focus rules cut "extra Red units alive" (how many more
Red units live than perfectly focused damage would leave) from 2.3 to 0.7, and won more often
(80% and 68% vs 63%). Retreating weak units is a losing tactic (15% wins) and doubles battle
length, which is why exp05 keeps retreat rare (section 8).

### What the simulator taught us (README "Findings")

- Units never shoot on their own: the policy must choose attack actions.
- Starting positions don't depend on the seed unless randomized; aggression 1.0 with a fixed
  spawn replays one game.
- Copying a running game with `deepcopy` diverges; replaying from seed and actions is exact.

---

## 4. The data layer (`sindy_wm.io`) and the experiments

- [`collect`](../src/sindy_wm/io/collect.py#L17) plays episodes with consecutive seeds and saves
  them: [`create_dataset`](../src/sindy_wm/io/storage.py#L78) →
  [`save_episode`](../src/sindy_wm/io/storage.py#L129) (snapshots, actions, controls, one line
  in `episodes.jsonl`) → [`finalize_dataset`](../src/sindy_wm/io/storage.py#L179) (checks every
  seed is there, marks it complete). With an [`EpisodeSampler`](../src/sindy_wm/envs/episode_sampling.py#L39)
  every episode draws its own conditions and records them in its index line.
- Reading back: [`load_manifest`](../src/sindy_wm/io/storage.py#L230),
  [`load_episode_index`](../src/sindy_wm/io/storage.py#L236),
  [`load_snapshots`](../src/sindy_wm/io/storage.py#L244),
  [`load_actions`](../src/sindy_wm/io/storage.py#L249),
  [`load_controls`](../src/sindy_wm/io/storage.py#L254).
- Summaries across an experiment: [`load_all_indexes`](../src/sindy_wm/evaluation/sweep.py#L16),
  [`summarize`](../src/sindy_wm/evaluation/sweep.py#L60) (win rate with a
  [Wilson interval](../src/sindy_wm/evaluation/sweep.py#L50), duration, distinct outcomes),
  [`heatmap`](../src/sindy_wm/evaluation/sweep.py#L80).

### The experiments

| | Scenario | Spawn | Per setting | What varies | What it was for |
|---|---|---|---|---|---|
| exp01 | 5v5 | fixed, 14.2 apart | 500 | aggression × step_mul grid | first look at outcomes |
| exp02 | 5v5 | random, 8–16 | 100 | same grid | does random spawn add variety? |
| exp03 | 10v10 | random, 8–16 | 500 | same grid | first SINDy fits; has engagement columns |
| exp04 | 10m map | sampled | 300 (pilot) | everything per battle | check the ranges for exp05 |
| exp05 | 10m map | sampled | 20,000 + 5,000 held out | everything per battle | the dataset for the thesis models |

Scripts: [`step_aggression_sweep.py`](../scripts/step_aggression_sweep.py) (exp01),
[`collect_data.py`](../scripts/collect_data.py) (exp02, exp03),
[`collect_pilot.py`](../scripts/collect_pilot.py) (exp04),
[`collect_sampled.py`](../scripts/collect_sampled.py) (exp05).

What the grids showed (notebooks 04 and 05):

- Blue's win rate rises with aggression and falls with `step_mul`: Red reacts every tick, Blue
  only every `step_mul` ticks, and Blue's random actions last a whole step.
- Battles last about 7.5–9.5 s at every `step_mul`: it changes the recording and Blue's reaction
  time, not the battle.
- Random spawns turn "one game repeated" into hundreds of distinct outcomes, and lower Blue's win
  rate mostly at `step_mul` 1–2. Distance alone doesn't explain it; the fixed exp01 layout seems
  to have been unusually good for Blue.

---

## 5. From snapshots to states (`sindy_wm.states.aggregate`)

SINDy needs a small state, not 20 units × 15 columns.

- [`level_a`](../src/sindy_wm/states/aggregate.py#L7): per team, `health` (hp + shield summed),
  `alive` (units alive), and `distance` between the teams' centres.
- [`engagement`](../src/sindy_wm/states/aggregate.py#L53): per team, `engaged` (units with a
  target in range, i.e. actually shooting), `targets` (distinct enemies shot at) and `speed`.

Three facts about these states shaped the modelling:

1. **Health is smooth-ish, alive is a step function.** Health drops one shot at a time; alive drops
   once per death. Derivatives of alive are mostly 0 with spikes.
2. **Alive is not just rounded health.** On exp03, alive equals ⌈10 × health⌉ only 23–33% of the
   time; on average 1.4–1.7 more units are alive than focused damage would leave, because each
   side spreads fire over about 2.5 targets. So alive carries information about *how damage is
   spread*, which matters for firepower.
3. **Health works as a clock.** On exp03, a side at full health has only 20% of its units in
   range; below 20% health, 92% are. High health means "early in the fight, units still walking
   in". Any model on health alone can use health as a proxy for time.

Distance was considered and dropped as a state: after trimming to the first damage (section 6),
the spread in starting distance collapses (std 2.10 → 0.43) and mostly decides *when* the fight
starts (correlation 0.98), which trimming removes.

---

## 6. Models (`sindy_wm.models`)

### 6.1 Preparing the data

- [`to_trajectory`](../src/sindy_wm/models/trajectories.py#L14): trims each episode to start one
  step before the first damage (the approach phase has no dynamics in health), downsamples, and
  restarts time at 0.
- [`derivative_data`](../src/sindy_wm/models/trajectories.py#L43): the derivative is estimated
  with a **Savitzky–Golay filter** (fit a local quadratic over `smooth_window` points, take its
  slope). It returns the *smoothed* states and their derivatives, exactly what `ps.SINDy` uses,
  so all three model types are fitted on identical data (a test checks this).

**Is the sawtooth a problem?** Each tooth is one shot, so the derivative is a pulse train. What
the model needs is the *average rate*, and that is well defined over half a second: in 0.25 s
windows 13% have no damage at all, in 0.5 s windows 1%, in 1 s none, and the mean rate is stable
at about −0.10/s. Smoothing estimates that average.

**Is stacking episodes correct?** Yes, if each episode is differentiated *before* stacking. The
fit is a sum over rows, so row order doesn't matter (shuffling all rows changed coefficients by
10⁻¹⁶). Differentiating *after* stacking turns the jump between one battle's end and the next
one's start into a huge fake derivative and gives nonsense coefficients. Caveats: rows within an
episode are correlated (judge uncertainty by resampling episodes, not rows), and the first and
last few points of each episode have noisier derivatives (dropping them moved Red's equation
noticeably on exp03).

### 6.2 Three model types, one interface

All three have `predict`, `simulate`, `coefficients`, `get_feature_names` and `equations`, so
everything downstream works on any of them.

| Model | Form | Fitted by |
|---|---|---|
| polynomial SINDy | $\dot X = \Theta(X)\,\Xi$, library $1, B, R, B^2, BR, R^2, \dots$ | [`fit_sindy`](../src/sindy_wm/models/sindy.py#L9): STLSQ (ridge regression, drop coefficients below `threshold`, repeat, refit) |
| Lanchester | $\dot B = -aR$, $\dot R = -bB$ | [`fit_lanchester`](../src/sindy_wm/models/lanchester.py#L54): closed-form least squares, $a = -\langle R,\dot B\rangle / \langle R,R\rangle$ |
| gated | $\dot B = -R\,p_B(B,R)$, $\dot R = -B\,p_R(R,B)$ | [`fit_gated`](../src/sindy_wm/models/gated.py#L90): STLSQ on a library where every term is multiplied by the opponent ([`GatedModel`](../src/sindy_wm/models/gated.py#L12)) |

**Why the gated model.** The polynomial fits produced terms like $+0.197\,R$ in $\dot R$: Red's
loss rate shrinks as Red weakens. Inside the data that reads as "fewer targets, more wasted
shots", but once Blue is dead it makes Red's health *grow* (visible in recap Part 6). In the gated
form every term contains the opponent, so a side cannot change once its opponent is gone: no
healing after a battle, by construction, while still allowing nonlinear "effective firepower"
$p(B, R)$. Degree 0 is exactly Lanchester.

**The physical scale.** A marine does 6 damage per 0.61 s and has 45 hp, so a full team firing
drains the other's normalized health at about 0.22/s. On exp03 (`10m_attack_a090_s1_v1`) the
fitted Lanchester $a, b$ = 0.160, 0.170 are 73% and 78% of that: the rest is lost to moving,
being out of range and overkill. Coefficients have a meaning.

### 6.3 Running a model forward

- [`simulate`](../src/sindy_wm/models/simulation.py#L14): integrate from a real state along an
  episode's own time points; returns `None` instead of crashing if the model blows up.
- [`simulate_to_elimination`](../src/sindy_wm/models/simulation.py#L31): integrate until a side
  reaches 0.02, using the ODE solver's *events* (stop exactly when `min(B, R) − 0.02` crosses 0,
  or when the model explodes). Checked against the closed-form square-law solution to 6 decimals.

---

## 7. How to judge a model: three generations of metrics

[`metrics.py`](../src/sindy_wm/models/metrics.py) holds all of them.

### First: who wins, at the real end time

[`evaluate`](../src/sindy_wm/models/metrics.py#L17) /
[`summarize_eval`](../src/sindy_wm/models/metrics.py#L85) simulate test battles from the start
and from midway, and originally called the side with more health at the *real* end time the
winner. That uses the real battle length, which a model wouldn't know.

### Second: who wins, simulating until elimination

The same functions now also run `simulate_to_elimination`: `winner_acc_*_elim` (did the model
eliminate the right side?), `duration_mae_*` (error in predicted remaining time),
`no_elimination`.

Neither version could beat the simple baseline "whoever has more health at midway wins", and it
turned out to be a **ceiling, not a failure**: on all 500 battles of `10m_attack_a090_s1_v1`, the
leader baseline gets 0.862; the best classifier using health gets 0.858, adding alive 0.862,
adding engagement 0.874. The remaining outcomes depend on things outside the state (positions,
random actions). And with 100 test battles, winner accuracy has about ±3.5 points of noise
(0.93 on the test split vs 0.86 on all data).

### Third: error a few seconds ahead (the world-model metric)

[`evaluate_horizons`](../src/sindy_wm/models/metrics.py#L119): from start points every second in
every test battle, integrate the model from the *real* state 0.5, 1, 2 and 4 s ahead and compare
with what happened. Two baselines on the same points: *persistence* (nothing changes) and
*last slope* (the last second's rate continues).

Two further checks:

- [`healing_share`](../src/sindy_wm/models/metrics.py#L203): share of states where a model makes a
  side's health increase (should be 0).
- **Stability**: refit on bootstrap resamples of the training *episodes*; a term whose spread is
  large or whose sign flips is not identified by the data (notebook 04 Part 6).

### What the third metric showed (exp03, `10m_attack_a090_s1_v1`, error 4 s ahead)

| Model | Terms | Error 4 s ahead | Heals somewhere |
|---|---|---|---|
| persistence | – | 0.393 | – |
| last slope | – | 0.092 | – |
| Lanchester | 2 | 0.091 | no |
| polynomial degree 1 | 3 | 0.090 | yes |
| polynomial degree 2, threshold 0.05 | **5** | **0.036** | yes (15% of states) |
| gated degree 1 | **6** | **0.039** | **no** |
| gated degree 2 | 10 | 0.037 | no |

- With the right metric, SINDy clearly does something: 2–3× smaller error than "keep going" at
  2–4 s, which the winner metrics completely hid.
- The 12-term degree-2 model was dense only because its threshold was too low: 5 terms predict as
  well.
- **Gated degree 1 is the best candidate so far**: nearly as accurate, never heals, and all six
  terms are stable (spread at most 0.017, no sign flips). Degree 2 adds unstable terms.

```
B' = −R · (0.408 − 0.150 B − 0.176 R)
R' = −B · (0.374 − 0.122 R − 0.154 B)
```

The negative terms say effective firepower per unit is lower when the shooting side is at full
health: the health-as-clock effect from section 5, i.e. engagement ramp-up in disguise, not a
combat law.

---

## 8. Why the data had to change, and how exp05 is designed

### What the old data couldn't tell the model

1. **Every battle ran down the diagonal.** All battles started at (1, 1) with equal teams; Blue
   and Red health correlate at 0.965. The model can't separate a Blue term from a Red term, so it
   trades them against each other (the source of odd self-terms and unstable coefficients).
2. **Health was a clock** (section 5): the extra terms model ramp-up, which won't transfer to
   battles that start differently.
3. **No variation in actions** within a dataset.
4. **Small test sets,** and no held-out set.

### The design

- **Per-episode sampling** ([`EpisodeSampler`](../src/sindy_wm/envs/episode_sampling.py#L39)):
  instead of a grid with hundreds of identical-setting battles per cell, every battle draws its
  own conditions. With many knobs this covers the space far better, and every battle adds new
  information. Summaries by condition are made afterwards by binning.
- **Uneven sides through the force ratio:** Red gets 6–14 units, and the **force ratio** (Blue's
  units ÷ Red's units) is drawn in 0.75–1.33, log-uniformly so Blue ahead by a third and Red
  ahead by a third are equally likely. It counts units, not health: firepower comes from the
  number of units (a marine deals full damage until it dies), staying power from health.
- **Damaged starts:** each side starts damaged with probability 0.5 (40–100% health). This breaks
  "full health = early", so the clock effect can be told apart from combat.
- **Starts in contact:** spawn distance 4–16, partly inside marine range (5), so some battles have
  no walk-in phase.
- **One policy setting per battle** (`SwitchingPolicy(segment_seconds=None)`): plain SINDy
  assumes each battle is one consistent system; the setting still varies across battles and is
  logged per step. Retreat is rare (8% and 5% of draws) because retreating battles last longer
  and would otherwise dominate the steps.
- **Separate random streams:** the policy uses `default_rng(seed)`, the policy schedule
  `[seed, 1]`, the sampler `[seed, 2]`, so one seed still reproduces everything.
- **Size and split:** 20,000 battles for fitting and choosing models, 5,000 with new seeds (from
  1,000,000) for final results only. `step_mul` 2 (0.125 s), the resolution the fits use.

### What the pilot and test runs showed

[`print_coverage`](../src/sindy_wm/evaluation/coverage.py#L21) checks a sampled dataset:

| Check | exp03 | exp05 test run (400 battles) |
|---|---|---|
| corr(Blue health, Red health), common scale | 0.97 | 0.28 |
| corr(Red health per unit, share of Red units in range) | −0.72 | −0.19 |
| Blue win rate by force ratio (≤0.8 … >1.25) | – | 6%, 20%, 49%, 74%, 95% |
| battles / steps with a retreat rule | – | 14% / 26% |

The diagonal and the clock are largely broken, and most battles are contested.

---

## 9. Open items before and after collecting

**Must do before fitting on exp05**

- **Common normalization.** [`level_a`](../src/sindy_wm/states/aggregate.py#L7) with
  `normalize=True` divides each side by its *own* starting value, so every side would start at 1
  and the uneven sides and damaged starts would vanish. Health needs a common reference (e.g. hp
  per marine × a reference count, as `print_coverage` already does).

**Worth doing in the analysis**

- Report coefficient stability (bootstrap by episode, or `ps.EnsembleOptimizer`) for every model.
- Consider dropping or down-weighting the noisy first/last points of each episode.
- Fit methods that avoid pointwise derivatives: a discrete-time map $x(t+\Delta) = f(x(t))$ at the
  planner's step, or weak SINDy. They also match the world-model use.
- Test whether the gated model's "clock" terms disappear on exp05; if they do, that's a clean
  result about engagement.
- Use alive or engagement as a firepower input in the gated form ($\dot H_B = -k \cdot$ alive$_R$
  was already found by an early run).

**Later datasets** (after exp05 has been fitted)

1. Scaling: 3–30 units per side, to test the square law's $N^2$ scaling and crowding.
2. An out-of-range test set (e.g. 16–20 units) for extrapolation.
3. Another unit type, both sides the same (e.g. stalkers), to test whether the model form carries
   over with different coefficients.
4. Models with control (SINDYc), with a dataset that switches settings within battles
   (`SwitchingPolicy` with `segment_seconds=(1, 3)`).

---

## 10. Where to look

| For | Go to |
|---|---|
| The whole pipeline, run once | [`notebooks/00_recap.ipynb`](../notebooks/00_recap.ipynb) |
| How the model code works | [`notebooks/01_models_explained.ipynb`](../notebooks/01_models_explained.ipynb) |
| Sweeps, first fits, metrics, sensitivity, sparsity vs. accuracy | [`notebooks/02_analysis/04_sweep_and_first_sindy.ipynb`](../notebooks/02_analysis/04_sweep_and_first_sindy.ipynb) |
| Fixed vs. random spawn | [`notebooks/02_analysis/05_compare_exp01_exp02.ipynb`](../notebooks/02_analysis/05_compare_exp01_exp02.ipynb) |
| Health vs. alive vs. engaged as Lanchester drivers | [`notebooks/02_analysis/06_lanchester_drivers.ipynb`](../notebooks/02_analysis/06_lanchester_drivers.ipynb) |
| Setup, structure, conventions, findings | [`README.md`](../README.md) |
| Earlier session notes | [`docs/`](.) |
