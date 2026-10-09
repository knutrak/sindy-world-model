# 2026-10-08 · Randomized spawn positions

Session notes for what was built today: randomizing the starting distance and
bearing between the two teams per episode, plus the tooling to analyse and compare the result
against the existing fixed-spawn experiment (`exp01`). This is a supplement to the README, not a
replacement — the README's Findings/Next steps sections were updated too; this doc is for the
function-level detail and the reasoning behind the choices.

## New files

### `src/sindy_wm/envs/spawn_sampling.py`

Pure geometry, no SMAClite import — easy to unit-test on its own.

- **`SpawnSample`** (dataclass): `ally_pos`, `enemy_pos`, `attack_point` — three `(x, y)` tuples.
  `attack_point` is included because SMAClite's built-in Red AI attack-moves toward a fixed point
  before it sees any Blue unit; it has to track the sampled Blue position or Red would walk
  toward the scenario's original, now-stale spawn instead.
- **`sample_opposing_spawns(rng, bounds, min_distance, max_distance, margin=1.5) -> SpawnSample`**
  Samples a random bearing, then a distance between the two group centers along it — both teams
  keep their own formation, only the separation and direction between them vary. Both group
  centers are placed around the fixed middle of `bounds` (never randomized itself — the terrain
  is open, so only the relative configuration affects the dynamics, not absolute position). This
  means the longest distance that fits a given bearing is computed directly with trig, so there's
  no search/rejection loop: *(simplified 2026-10-09 — it originally guessed a random center too
  and retried up to `max_tries` times until both groups landed in bounds; replaced once it became
  clear the center didn't need to vary at all)*. Raises `ValueError` up front if `margin` alone
  leaves no room in `bounds`, or if `min_distance` can't fit at every bearing.

  ```python
  from sindy_wm.envs.spawn_sampling import sample_opposing_spawns
  import numpy as np

  spawn = sample_opposing_spawns(
      np.random.default_rng(0), bounds=(0, 31, 8, 23), min_distance=8, max_distance=16
  )
  spawn.ally_pos, spawn.enemy_pos, spawn.attack_point
  ```

### `src/sindy_wm/evaluation/sweep.py`

Extracted from notebook `04`'s Part 1 (it was about to be needed a second time, by notebook `05`
— the project's own stated rule for promoting notebook code to `src/`).

- **`load_all_indexes(experiment_dir) -> pd.DataFrame`** — one row per episode across every
  *complete* dataset in an experiment folder, with each dataset's manifest parameters
  (`step_mul`, `aggression`, `policy`) joined in and duration converted to seconds. Skips
  anything not `complete` (prints which).
- **`wilson_interval(wins, n, z=1.96) -> (low, high)`** — 95% CI for a win rate, works near 0/1.
- **`summarize(group) -> pd.Series`** — win rate (+ interval), duration, and `distinct_outcomes`
  for one dataset's episodes. Meant to be used with `episodes.groupby([...]).apply(summarize)`.
- **`heatmap(summary, value, title, cmap="RdBu", ax=None) -> (fig, ax, pivoted_table)`** —
  aggression × step_mul heatmap of one summary column. Pass `ax` to draw into an existing subplot
  (added for notebook `05`'s side-by-side comparison); omit it to get its own figure, same as
  notebook `04` already did.
- **`episodes_by_time(states, episode_ids) -> list[pd.DataFrame]`** — one Level-A table per
  episode, indexed by time in seconds, for `plot_many_episodes`.

### `notebooks/02_analysis/05_compare_exp01_exp02.ipynb`

Loads both experiments via `sweep.py`, and shows: win-rate heatmaps side by side plus their
difference; a `distinct_outcomes` comparison (the clearest "did randomizing spawn actually work"
check); mean battle duration by `step_mul`; and a full merged table with win-rate deltas. Just
open and run top to bottom — it discovers both experiment folders under `RAW_DIR` by name.
Validated by executing all cells against the real `exp01`/`exp02` data on disk (see the Findings
numbers in the README — they came from this notebook).

### Tests

- `tests/test_spawn_sampling.py` — bounds/distance are respected, same `rng` gives the same
  sample, and a too-tight `margin` raises.
- `tests/test_randomized_spawn.py` — `collect(..., randomize_spawn=True)` actually produces
  different step-0 positions per episode, and the same seeds reproduce exactly on a second run.
- `tests/test_sweep.py` — `load_all_indexes` skips incomplete datasets and joins manifest
  parameters correctly; `summarize`/`wilson_interval` sanity checks.

## Modified files

- **`src/sindy_wm/envs/scenarios.py`**: added `seconds_per_step(step_mul)` (factored out of
  `StepMulWrapper`), `load_map_info(scenario)`, `walkable_bounds(map_info)` (bounding box of
  `NORMAL` terrain — for `5m_vs_5m`'s `SIMPLE` preset this is `x∈[0,31], y∈[8,23]`, *not* the full
  nominal 32×32), and `_with_spawn(map_info, spawn)` (returns a copy of `map_info` with the
  ALLY/ENEMY group centers and `attack_point` overridden — raises `NotImplementedError` if the
  scenario doesn't have exactly one group per faction). `make_env(...)` gained a `spawn=` kwarg
  that routes through these.
- **`src/sindy_wm/data/collect.py`**: `collect(...)` gained `randomize_spawn` and
  `spawn_distance_range`. When enabled, it builds one environment per episode (via
  `make_env(..., spawn=...)`) instead of one per dataset, because SMAClite bakes group positions
  into the environment at construction time, not at `reset()`. `dataset_config(...)` records the
  distance range and the sampling scheme in the manifest's `scenario.spawn_randomization`.
- **`.gitignore`**: added `.DS_Store`; untracked the three copies already committed
  (`.DS_Store`, `src/.DS_Store`, `src/sindy_wm/.DS_Store`) — they were making `git_dirty` true on
  every run regardless of real code changes, which would have blocked `ALLOW_DIRTY = False`
  thesis collection for no reason.

## How to use it

Collect a dataset with randomized spawn (see the README's "Randomized starting positions"
section for the full explanation):

```python
from sindy_wm.data.collect import collect
from sindy_wm.envs.policies import AttackNearestPolicy

collect(
    dataset_dir=my_dataset_dir,
    policy=AttackNearestPolicy(aggression=0.8),
    description="...",
    randomize_spawn=True,
    spawn_distance_range=(8.0, 16.0),
)
```

Analyse one experiment the way notebook `04` does, or compare two experiments with notebook `05`
— both just need the experiment folder name(s) under `data/raw/`.

## Design decisions (and why)

- **Formation preserved, only distance/bearing randomized** — not full independent per-unit
  scatter. Keeps `trim_to_engagement` valid (there's still one clean approach-then-fight
  structure) and avoids degenerate point-blank or terrain-edge spawns. Full scatter was discussed
  and deliberately deferred: it would need bypassing SMAClite's `Group` placement API (mutating
  live `Unit.pos` after `reset()` and manually refreshing SMAClite's neighbour-finder caches,
  which only self-refresh at the end of each world-tick) and would likely need rethinking
  `trim_to_engagement` itself.
- **The spawn sampler uses the episode's plain `seed`** (`np.random.default_rng(seed)`), the same
  one used to seed the env and the policy — a deliberate simplicity choice so one number still
  describes a whole episode, rather than inventing a second derivation rule. Consequence: the
  spawn draw and e.g. the policy's first random decision start from the identical underlying
  PCG64 stream (verified: `default_rng(7).uniform(8,16)` and a separate `default_rng(7).random()`
  share the same raw draw). Fully deterministic and reproducible either way; just not
  statistically independent. Already true of the env/policy relationship before today.
- **`x, y` (and even plain `distance`) were not added to the SINDy state** — checked empirically,
  not just argued: `trim_to_engagement` collapses almost all the spawn-distance variation before
  a trajectory ever reaches SINDy (std 2.19 at spawn → std 0.46 at the trimmed start → barely
  moves after that). See the README's Findings for the numbers and the `corr = 0.97` between
  spawn distance and *when* engagement starts, which is where that variation actually goes.

## Known gaps / left for later

- Notebook `04` itself was **not** refactored to import from `sweep.py` — it was being actively
  edited in a live kernel during this session (changing `EXPERIMENT`, then `DATASET`, while this
  work was in progress), so touching it risked a lost-update race with the next autosave. The
  mechanical swap (replace the inline `def load_all_indexes...` etc. with
  `from sindy_wm.evaluation.sweep import ...`) is still worth doing once it's not mid-edit.
- `save_run`'s `config.json` (still in notebook `04`, not yet moved to `src/`) doesn't record
  *which experiment* a dataset came from — only the bare dataset name, which `exp01` and `exp02`
  both reuse. Add `"experiment": EXPERIMENT` to `run_config["dataset"]` before relying on saved
  runs to compare SINDy coefficients across experiments.
- The exp01-vs-exp02 win-rate drop (see README Findings) isn't fully explained — distance alone
  is ruled out, approach-angle is only weakly suggestive. A dataset that randomizes bearing only,
  at a fixed distance, would isolate it cleanly.
- `NumpyVelocityUpdater`'s `"Using the numpy RVO2 port"` print now fires once per *episode*
  instead of once per *dataset* (confirmed: one new environment is built per episode when
  `randomize_spawn=True`). Confirmed harmless (~0.4 ms per construction) and left alone; trivial
  to suppress with `contextlib.redirect_stdout` around the per-episode `make_env` call if the
  noise becomes annoying.
- Map resizing (for bigger battles) was discussed but not implemented: unnecessary for unit
  count — SMAClite's own `bane_vs_bane` (24v24) runs on the same 32×32 `SIMPLE` terrain — and
  would need a custom terrain grid (not a `terrain_preset`, which are all fixed 32×32) if ever
  needed for extra spawn-position headroom instead.
