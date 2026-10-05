# SINDy world models for multi-agent combat

Master's thesis project (NTNU / FFI, 2026–2027). The goal is to learn compact,
interpretable models of combat dynamics from simulation data using sparse system
identification (SINDy), and later to use these models as world models for
planning and decision-making.

The current focus is **Phase 1: data ingestion and analysis** in
[SMAClite](https://github.com/uoe-agents/smaclite), a lightweight Python
reimplementation of the StarCraft Multi-Agent Challenge (SMAC).

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
- In notebooks, select the project's `.venv` as the kernel.

## Project structure

```
src/sindy_wm/            reusable code, imported by notebooks and scripts
├── paths.py             standard project paths (PROJECT_ROOT, RAW_DIR, ...)
├── envs/                everything that talks to SMAClite
│   ├── scenarios.py     make_env(): create environments by scenario name
│   ├── policies.py      RandomPolicy, AttackNearestPolicy
│   ├── episode.py       play_episode(): play and record one episode
│   └── smaclite_snapshot.py   SnapshotLogger: per-unit state tables
├── data/
│   └── storage.py       save_episode(), load_episode()
├── states/
│   └── aggregate.py     level_a(): team-level state from snapshots
├── models/              (empty: SINDy models come next)
└── evaluation/
    └── plots.py         plot_level_a, plot_many_episodes, plot_positions, animate_positions
notebooks/               exploration (01_ingestion, 02_analysis, 03_model_exploration)
scripts/                 command-line runs
configs/scenarios/       custom SMAClite scenarios (e.g. 5m_vs_5m.json)
data/                    datasets (not in git): raw/ is read-only, processed/ is disposable
runs/                    experiment outputs (not in git)
tests/                   tests (to be written)
```

## Quick start

```python
from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.policies import AttackNearestPolicy
from sindy_wm.envs.scenarios import make_env
from sindy_wm.evaluation.plots import plot_level_a
from sindy_wm.states.aggregate import level_a

env = make_env("5m_vs_5m")  # built-in name or file in configs/scenarios/
episode = play_episode(env, AttackNearestPolicy(aggression=0.8), seed=0)

states = level_a(episode.snapshots)  # one row per step
plot_level_a(states, title="5m_vs_5m, seed 0")
```

In notebooks, set `plt.rcParams["animation.html"] = "jshtml"` so that
`animate_positions(episode.snapshots)` plays inline.

## Data model

- **Episode**: one game, from `env.reset()` until one side is eliminated or
  `max_steps` is reached.
- **Snapshot**: the state of every unit at one step (team, unit id, type,
  position, hp, shield, cooldown, alive). Stored as one Parquet file per episode.
- **State**: a compressed summary computed *from* snapshots. Level A is fully
  aggregated: `health_blue`, `health_red`, `alive_blue`, `alive_red`, `distance`
  (between the centroids of alive units).

Principle: **log raw snapshots once, derive everything else.** States are never
stored during collection, so state definitions can change without replaying games.

## Conventions

- Only `envs/` imports `smaclite`. Everything downstream works on snapshot tables.
- Prototype in notebooks; move a function to `src/` when it is needed a second time,
  or immediately if it produces saved data.
- Every episode is seeded; the same seed drives both the environment and the policy.
- Raw datasets are never overwritten: use a new dataset name instead.
- Ruff for linting and formatting (`uv run ruff check .`, `uv run ruff format .`).

## Findings about SMAClite so far

- **Dead units are removed** from `env.unwrapped.agents` / `.enemies`. The
  `SnapshotLogger` records the full roster at reset and keeps dead units with `hp = 0`.
- **No time limit**: SMAClite never truncates episodes, so `play_episode` has its
  own `max_steps` (default 200).
- **step_mul** (game ticks per step, 16 ticks = 1 s) is a global constant in SMAClite.
  `make_env(..., step_mul=k)` wraps the environment so each one keeps its own value.
  Default 8 = 0.5 s per step, as in SMAC.
- **Starting positions do not depend on the seed**: every episode of a scenario
  starts from the same state.
- **Units do not fire back on their own**: policies must choose attack actions explicitly.
- **Copying the game with `deepcopy` diverges**; replaying from seed plus action log
  reproduces episodes exactly.
- **Blue loses mirror fights at step_mul 8**, even with `AttackNearestPolicy`, apparently
  because Red's AI reacts every tick while Blue decides every 8 ticks. With step_mul 1,
  Blue wins 5v5 marines consistently.

## Next steps

- Extend `storage.py` with a dataset manifest (scenario, policy and parameters,
  step_mul, seeds, outcomes, git commit) and save actions.
- `scripts/collect_data.py` for reproducible dataset generation.
- More varied data: scenario variants, smaller step_mul, retreating policies.
- Tests for snapshot logging, reproducibility and storage.
- First SINDy fit on Level A for 5m_vs_5m.