# SINDy world models for multi-agent combat

Master's thesis project (NTNU / FFI, 2026–2027). The goal is to learn compact,
interpretable models of combat dynamics from simulation data using sparse system
identification (SINDy), and later to use these models as world models for
planning and decision-making.

The current focus is **Phase 1: data ingestion and analysis** in
[SMAClite](https://github.com/uoe-agents/smaclite), a lightweight Python
reimplementation of the StarCraft Multi-Agent Challenge (SMAC). Reproducible data
collection is in place, and a first SINDy model has been fitted on team-level states.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the project root:

```bash
uv sync                      # create .venv with all dependencies
uv run nbstripout --install  # strip notebook outputs on commit (once per clone)
```

Notes:

- SMAClite is installed from GitHub, pinned to a specific commit (see `pyproject.toml`).
- SMAClite pins `Rtree==1.0.0`, which fails to load on recent Python versions.
  `pyproject.toml` overrides this with `rtree>=1.1`.
- Models use PySINDy 2.x. Notebooks that format tables with `DataFrame.style` need
  `jinja2` (`uv add --dev jinja2`).
- Notebook 04's interactive slider (Part 5) needs `ipywidgets`, already a dev dependency.
- In notebooks, select the project's `.venv` as the kernel.

## Project structure

```
src/sindy_wm/            reusable code, imported by notebooks and scripts
├── paths.py             standard project paths (PROJECT_ROOT, RAW_DIR, SCENARIO_DIR, ...)
├── provenance.py        code_info(), file_sha256(): which code and inputs produced a result
├── envs/                everything that talks to SMAClite
│   ├── scenarios.py     make_env(): create environments by scenario name, optionally with
│   │                    a randomized spawn; walkable_bounds(), load_map_info()
│   ├── spawn_sampling.py   sample_opposing_spawns(): random distance/bearing between two groups
│   ├── policies.py      RandomPolicy, AttackNearestPolicy (each describes itself via config())
│   ├── episode.py       play_episode(): play and record one episode
│   └── smaclite_snapshot.py   SnapshotLogger: per-unit state tables
├── data/
│   ├── storage.py       datasets: create, save episodes, finalize, load
│   └── collect.py       collect(): play episodes with a policy and save them as a dataset,
│                        optionally with a randomized spawn (see Collecting datasets below)
├── states/
│   └── aggregate.py     level_a(): team-level state from snapshots; engagement(): engaged/
│                        targets/speed per team, needs snapshots with target_*/vx/vy columns
├── models/
│   └── sindy.py         to_trajectory, fit_sindy, simulate, evaluate, summarize_eval,
│                        plot_predictions, save_run: moved from notebook 04 once settled;
│                        fit_lanchester, LanchesterModel: the classical 2-parameter square
│                        law, as an explicit (not SINDy-sparsified) baseline
└── evaluation/
    ├── plots.py         plot_level_a, plot_many_episodes, plot_positions, animate_positions
    └── sweep.py         load_all_indexes, summarize, heatmap: condense an experiment folder
                         (many datasets) into one table and compare win rates/duration
notebooks/               exploration (01_ingestion, 02_analysis, 03_model_exploration,
                         04_sweep_and_first_sindy, 05_compare_exp01_exp02,
                         06_lanchester_drivers)
scripts/
├── step_aggression_sweep.py   collect exp01: fixed spawn, aggression x step_mul grid
└── collect_data.py            collect another experiment's datasets; existing ones are skipped
configs/scenarios/       custom SMAClite scenarios (5m_vs_5m.json, 10m_vs_10m.json)
data/                    datasets (not in git): raw/ is read-only, processed/ is disposable
runs/                    experiment outputs, e.g. SINDy fits (not in git: back it up)
docs/                    session notes on larger additions (not code documentation; see src/
                         docstrings for that)
tests/                   pytest suite: storage, reproducibility, snapshot logging, spawn
                         sampling, sweep summaries (`uv run pytest`)
```

## Quick start

Play one episode and look at it:

```python
from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.policies import AttackNearestPolicy
from sindy_wm.envs.scenarios import make_env
from sindy_wm.evaluation.plots import plot_level_a
from sindy_wm.states.aggregate import level_a

env = make_env("5m_vs_5m", step_mul=8)  # built-in name or file in configs/scenarios/
episode = play_episode(env, AttackNearestPolicy(aggression=0.8), seed=0)

states = level_a(episode.snapshots)  # one row per step
plot_level_a(states, title="5m_vs_5m, seed 0")
```

In notebooks, set `plt.rcParams["animation.html"] = "jshtml"` so that
`animate_positions(episode.snapshots)` plays inline.

## Collecting datasets

Each experiment gets its own script in `scripts/` (e.g. `step_aggression_sweep.py` for exp01),
listing its datasets in `DATASETS` at the top (grids of parameters can be generated with
`itertools.product`). `collect_data.py` is the one currently being edited for the next
experiment. Run with:

```bash
uv run scripts/step_aggression_sweep.py   # or whichever experiment script
```

Datasets that already exist are skipped, so the script can be rerun after adding
entries or after a crash (delete an incomplete dataset folder first). Set
`ALLOW_DIRTY = False` and commit all changes before collecting datasets for the thesis,
so that the recorded git commit describes the code exactly.

Each dataset is a folder in `data/raw/`:

```
data/raw/5m_attack_a070_s1_v1/
    manifest.json        how the dataset was made: scenario (+ file hash), step_mul,
                         seconds_per_step, max_steps, policy and parameters, seeds,
                         git commit and dirty flag, package versions, status
    episodes.jsonl       one line per episode: seed, n_steps, battle_won, truncated,
                         final alive units and hp per team
    snapshots/episode_00000.parquet ...   per-unit state at every step
    actions/episode_00000.parquet ...     Blue's action per unit at every step
```

### Randomized starting positions

By default, a dataset uses the scenario's fixed group positions (every episode starts from the
same state). Pass `randomize_spawn=True` to `collect()` to instead sample a new distance and
bearing between the two groups every episode:

```python
collect(
    dataset_dir=...,
    policy=AttackNearestPolicy(aggression=0.8),
    description=...,
    randomize_spawn=True,
    spawn_distance_range=(8.0, 16.0),  # map units; the fixed 5m_vs_5m distance is 14
)
```

Both teams keep their own formation; only the distance and the direction between the two group
centers vary (see `envs.spawn_sampling.sample_opposing_spawns`). This needs a scenario with
exactly one ALLY and one ENEMY group (true for `5m_vs_5m`). The spawn sampler uses the episode's
own `seed` (same as the env and the policy), so one seed still describes the whole episode; the
realized positions end up in each episode's step-0 snapshot, nothing extra is recorded.
Individual units are *not* scattered independently (no formation at all) — that would need
reaching past SMAClite's supported placement API and would break `trim_to_engagement`
preprocessing (see Findings below); considered but intentionally not implemented.

Reading a dataset back:

```python
from sindy_wm.data import storage

manifest = storage.load_manifest(dataset_dir)
dt = manifest["config"]["env"]["seconds_per_step"]
index = storage.load_episode_index(dataset_dir)  # one row per episode
snaps = storage.load_snapshots(dataset_dir)  # all episodes, with episode_idx
states = snaps.groupby("episode_idx").apply(level_a, include_groups=False)
```

`states` then has a two-level index `(episode_idx, step)`. Select one episode with
`states.loc[3]`, which drops the episode level.

## Data model

- **Episode**: one game, from `env.reset()` until one side is eliminated or
  `max_steps` is reached.
- **Snapshot**: the state of every unit at one step (team, unit id, type,
  position, hp, shield, cooldown, alive).
- **Actions**: Blue's action per unit per step. The action at step t is chosen from
  the snapshot at step t. Together with the seed, actions make an episode replayable,
  and they are the control input for models with control.
- **Dataset**: many episodes collected with one configuration (scenario, step_mul,
  policy and parameters), described by its manifest.
- **State**: a compressed summary computed *from* snapshots. Level A is fully
  aggregated: `health_blue`, `health_red`, `alive_blue`, `alive_red`, `distance`
  (between the centroids of alive units).
- **Run**: one model fitted and evaluated with one set of choices, saved in `runs/`.

Principle: **log raw snapshots once, derive everything else.** States are never
stored during collection, so state definitions can change without replaying games.

## Runs

Each run is a new folder `runs/<date>_<time>_<short name>/`:

```
config.json        dataset (name and creation time), train/test episodes,
                   preprocessing and model choices, code version
model.json         state names, feature names, coefficients, equations
equations.txt      the model in readable form
metrics.json       summary metrics on test episodes
per_episode.csv    metrics per test episode
figures/           plots of predictions against data
```

Notebooks are for exploring; a result becomes a run once it is worth keeping.
Runs are never overwritten. `runs/` is not in git, so keep it in a backed-up location.

## Conventions

- Only `envs/` imports `smaclite`. Everything downstream works on snapshot tables.
- Prototype in notebooks; move a function to `src/` when it is needed a second time,
  or immediately if it produces saved data.
- Every episode is seeded; the same seed drives both the environment and the policy.
- Raw datasets and runs are never overwritten: use a new name instead.
- Compare across step_mul values in **seconds**, not steps.
- Give SINDy **one trajectory per episode**, never episodes stacked into one array.
- Split training and test data **by episode**. Keep a separate dataset with new seeds
  for final results, never used for choosing models.
- Ruff for linting and formatting (`uv run ruff check .`, `uv run ruff format .`).

## Findings so far

### About SMAClite

- **Dead units are removed** from `env.unwrapped.agents` / `.enemies`. The
  `SnapshotLogger` records the full roster at reset and keeps dead units with `hp = 0`.
- **No time limit**: SMAClite never truncates episodes, so `play_episode` has its
  own `max_steps`. When step_mul varies, set the limit in seconds and convert
  (`max_steps = seconds * 16 // step_mul`).
- **step_mul** (game ticks per step, 16 ticks = 1 s) is a global constant in SMAClite.
  `make_env(..., step_mul=k)` wraps the environment so each one keeps its own value.
  Default 8 = 0.5 s per step, as in SMAC.
- **Starting positions do not depend on the seed by default**: every episode of a scenario
  starts from the same state, unless `randomize_spawn=True` (see Collecting datasets above).
  Positions are baked into the environment at construction time, not at `reset()`, so a
  randomized-spawn dataset builds one environment per episode instead of one per dataset
  (harmless: ~0.4 ms each, but it does mean SMAClite's own "Using the numpy RVO2 port" print
  shows up once per episode instead of once per dataset).
- **Units do not fire back on their own**: policies must choose attack actions explicitly.
- **Copying the game with `deepcopy` diverges**; replaying from seed plus action log
  reproduces episodes exactly.

### From the 5m_vs_5m sweep (AttackNearestPolicy, 100 episodes per setting)

Blue win rate by aggression and step_mul:

| aggression | step_mul 1 | step_mul 2 | step_mul 4 | step_mul 8 |
|---|---|---|---|---|
| 0.5 | 25% | 8% | 1% | 0% |
| 0.7 | 65% | 38% | 12% | 3% |
| 0.9 | 89% | 71% | 35% | 3% |
| 1.0 | 100%* | 0%* | 0%* | 0%* |

\* Essentially one game repeated (see below), so not a rate.

- **Win rate rises with aggression and falls with step_mul.** Red's AI reacts every
  tick, while Blue decides every step_mul ticks. A likely reason the effect is larger at
  low aggression: Blue's random actions (probability 1 − aggression) last a whole step,
  1/16 s at step_mul 1 but 0.5 s at step_mul 8.
- **Battles last about 8–9 s at every step_mul.** step_mul changes how finely a battle is
  recorded and how often Blue decides, not the battle itself. At step_mul 8 a battle is
  only about 17 points long; at step_mul 1 about 140.
- **Aggression 1.0 gives (nearly) identical episodes**: the policy uses no randomness and
  every episode starts from the same state. Use aggression below 1 for varied data.
- With 100 episodes, a win rate near 50% has a 95% interval of about ±10 percentage
  points; smaller differences are not meaningful.
- **Balanced settings** (both teams win sometimes): `a070_s1`, `a070_s2`, `a090_s2`, `a090_s4`.

### From comparing exp01 (fixed spawn) vs exp02 (randomized spawn, 500 episodes per setting)

- **Randomizing spawn distance fixes the `aggression=1.0` column.** In exp01 it was ~1-2 distinct
  outcomes per dataset (one game repeated, not a rate: see above). In exp02, `distinct_outcomes`
  jumps to 300-450 across the whole grid, since initial separation is now a real source of
  variety even when the policy itself has none.
- **Blue's win rate is generally lower in exp02**, but not uniformly: the drop is concentrated at
  `step_mul` 1-2 (exp01: 25-100% → exp02: 13-60% across aggressions), which was also exp01's
  *best* regime for Blue. At `step_mul` 4-8, where exp01 was already near a floor, exp02's rates
  are flat or higher.
  - Checked and ruled out as the explanation: distance alone. Even exp02 episodes that happen to
    spawn at ~14-15 (matching exp01 exactly) only win ~60-64%, well below exp01's 86% at that
    setting (`a090_s1`).
  - Checked and inconclusive: approach bearing (Blue can only move in 4 cardinal directions per
    decision; Red's built-in AI moves continuously via RVO2). Correlation with win rate is
    essentially zero overall; a small near-distance-matched subsample hints it might matter
    (22% vs. 60% win rate, aligned vs. diagonal bearing) but n=9, too small to trust.
  - Best current read: the fixed `(9,16)`/`(23,16)` layout — same y, attack_point exactly on
    Blue's spawn, dead-on approach — was an atypically Blue-favorable corner of the configuration
    space, not a neutral baseline. exp02's numbers are probably the more representative picture.
    Not fully isolated; a dataset that randomizes bearing only, at a fixed distance, would cleanly
    separate the angle effect from distance (they're confounded in exp02 as collected).
- **x, y (or plain `distance`) should not be added as a SINDy-fit state.** `trim_to_engagement`
  collapses almost all of the variation `randomize_spawn` introduces before a trajectory ever
  reaches SINDy: spawn distance has std 2.19 across episodes, but distance at the trimmed
  engagement start has std only 0.46, and barely moves after that (mean range ~2.3 units for the
  rest of the trajectory). The information about initial separation ends up almost entirely in
  *when* engagement starts (`corr(spawn_distance, trim_start_step) = 0.97`), which trimming
  discards by re-zeroing every trajectory's clock. Raw `x, y` would be worse again: the dynamics
  are translation-invariant (open terrain), so only the relative configuration matters, not
  absolute position.

### About modelling

- **Health changes in discrete chunks** (one shot at a time). At fine time resolution
  this gives a staircase with mostly zero derivatives; smoothing or downsampling over
  several shots is needed. Recording at step_mul 1 and smoothing afterwards keeps the
  choice of time scale open.
- **Before the first shot, health does not change** while units approach each other.
  Trajectories are trimmed to start at the engagement.
- **From a common starting state, a deterministic model predicts the same battle every
  time**, i.e. the average fight. Predicting the winner from the start therefore cannot
  beat always guessing the more common outcome. Evaluation from midway, compared with
  the baseline "whoever has more health wins", is the more informative test.

### Which quantity should a Lanchester fit use? (notebook 06, `10m_vs_10m`, `exp03`)

Classical Lanchester theory models unit counts, not aggregate health — fit the *exact* square law
(`fit_lanchester`, no constant, no self-term, not a SINDy fit that happened to sparsify to it) on
`health`, `alive`, and the new `engaged` (from `engagement()`) as candidate drivers:

- **`health` and `alive` both land exactly on their own `leader_baseline_midway`** (0.89 and 0.71
  respectively) and are stable across a `SMOOTH_WINDOW` sweep (5-17): legitimate, trustworthy fits
  of the classical form, just not ones that beat the naive "whoever's ahead wins" rule.
- **This isn't a coincidence of these two drivers**: with `a, b > 0`, the square law's own
  structure (both quantities only decrease, faster when the opponent is currently stronger) makes
  a lead reversal from the midpoint rare by construction — matching `leader_baseline_midway` is
  close to what *any* driver's classical fit can do, not evidence of a good or bad fit on its own.
- **`engaged` behaves differently**: `b` comes out consistently *negative* across every window
  tested, not just noisy. Likely not a smoothing problem but the wrong *kind* of variable for this
  functional form — `engaged_*` can go up or down as units move in and out of range, so it's a
  fluctuating activity measure, not a monotonically depleting stock like health or alive count.
- **The general SINDy fit on health (degree 2, notebook 04) does beat `leader_baseline_midway`**
  (0.84 vs. 0.73, a different dataset) precisely because it isn't restricted to pure cross-coupling
  — it can pick up self-terms and quadratic terms the classical law excludes by construction. That
  comparison, not "does it look like Lanchester," is the real test of whether SINDy finds something
  beyond classical theory here.

## Next steps

- Interpret the first SINDy results (notebook 04) and record them here.
- Run notebook 06's driver comparison on more `exp03` settings, not just one dataset, to see if
  the `health`/`alive`/`engaged` ranking holds generally or is specific to that aggression/step_mul.
- Re-collect a 5v5 dataset with the new `vx`/`vy`/`target_*` snapshot columns, to compare driver
  behavior across scenario sizes — classical theory predicts `a`/`b` should be roughly invariant to
  force size under the square law; notebook 06 only has one scale (`10m_vs_10m`) to check so far.
- Isolate the exp01 vs exp02 win-rate drop: a dataset that randomizes bearing only, at the fixed
  distance of 14, would separate the angle effect from distance (see Findings above).
- Starting states that differ: scenario variants (e.g. 3v3, 5v3, 8v5).
- A richer state for SINDy: *not* `x, y` or `distance` (see Findings above); maybe a model per
  phase (approach, then fight) instead, now that `randomize_spawn` gives real variation in the
  approach phase to model.
- Intra-team jitter (looser formations, still two clustered sides) considered as a follow-up to
  `randomize_spawn`; deferred for now. Fully independent per-unit scatter (no team clustering at
  all) would need bypassing SMAClite's `Group`/`MapInfo` placement API and would break
  `trim_to_engagement`; treat as a deliberate stress test later, not a mainline dataset.
- A held-out dataset with new seeds for final evaluation.
- Later: retreating policies, and models with control (actions as input).