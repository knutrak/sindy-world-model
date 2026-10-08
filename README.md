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
- In notebooks, select the project's `.venv` as the kernel.

## Project structure

```
src/sindy_wm/            reusable code, imported by notebooks and scripts
├── paths.py             standard project paths (PROJECT_ROOT, RAW_DIR, SCENARIO_DIR, ...)
├── provenance.py        code_info(), file_sha256(): which code and inputs produced a result
├── envs/                everything that talks to SMAClite
│   ├── scenarios.py     make_env(): create environments by scenario name
│   ├── policies.py      RandomPolicy, AttackNearestPolicy (each describes itself via config())
│   ├── episode.py       play_episode(): play and record one episode
│   └── smaclite_snapshot.py   SnapshotLogger: per-unit state tables
├── data/
│   ├── storage.py       datasets: create, save episodes, finalize, load
│   └── collect.py       collect(): play episodes with a policy and save them as a dataset
├── states/
│   └── aggregate.py     level_a(): team-level state from snapshots
├── models/              (SINDy code moves here from notebooks when it settles)
└── evaluation/
    └── plots.py         plot_level_a, plot_many_episodes, plot_positions, animate_positions
notebooks/               exploration (01_ingestion, 02_analysis, 03_model_exploration,
                         04_sweep_and_first_sindy)
scripts/
└── collect_data.py      collect the datasets listed in DATASETS; existing ones are skipped
configs/scenarios/       custom SMAClite scenarios (e.g. 5m_vs_5m.json)
data/                    datasets (not in git): raw/ is read-only, processed/ is disposable
runs/                    experiment outputs, e.g. SINDy fits (not in git: back it up)
tests/                   tests (to be written)
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

List the datasets to collect in `DATASETS` at the top of `scripts/collect_data.py`
(grids of parameters can be generated with `itertools.product`), then run:

```bash
uv run scripts/collect_data.py
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
- **Starting positions do not depend on the seed**: every episode of a scenario
  starts from the same state.
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

## Next steps

- Interpret the first SINDy results (notebook 04) and record them here.
- Tests for snapshot logging, reproducibility and storage (needs a `replay_episode()`
  function in `envs/episode.py`).
- Move `save_run` and `load_all_indexes` from notebook 04 to `src/`.
- Starting states that differ: scenario variants (e.g. 3v3, 5v3, 8v5).
- Richer state for SINDy: add `distance` or alive counts; consider separate models for
  the approach and the fight.
- A held-out dataset with new seeds for final evaluation.
- Later: retreating policies, and models with control (actions as input).