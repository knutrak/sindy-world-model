# 2026-10-09 · Moved notebook 04's SINDy functions to src/

Notebook 04 (`04_sweep_and_first_sindy.ipynb`) defined its Part 3 (model fitting/evaluation)
functions inline, closing over notebook globals (`STATE_COLUMNS`, `trajectories`, `dt_fit`, ...).
Part 1's functions were already extracted to `evaluation/sweep.py` in an earlier session, but
notebook 04 itself still defined its own copies rather than importing them. Both are now fixed:
notebook 04 imports everything from `src/`, and notebook 05 already did.

## New file: `src/sindy_wm/models/sindy.py`

Moved from notebook 04's Part 3. Each function lost its dependence on notebook globals — they now
take `state_columns`, `trajectories`, `battle_won`, etc. explicitly instead of closing over them.

- **`to_trajectory(ep, state_columns, trim_to_engagement, downsample) -> pd.DataFrame`** — trims
  one episode's Level A table to start one step before the first damage, downsamples, and resets
  `t` to start at 0.
- **`fit_sindy(trajectories, train_ids, state_columns, dt, smooth_window, degree, threshold) -> ps.SINDy`**
  — fits a polynomial-library, STLSQ-sparsified SINDy model.
- **`simulate(model, traj, state_columns, start=0) -> np.ndarray | None`** — integrates the model
  from row `start` onward; `None` if integration fails or blows up, rather than raising.
- **`evaluate(model, ids, trajectories, state_columns, battle_won) -> pd.DataFrame`** — RMSE and
  winner-accuracy per episode, simulating from the start and from midway, against baselines.
- **`summarize_eval(model, ids, trajectories, state_columns, battle_won) -> dict`** — `evaluate`,
  averaged.
- **`plot_predictions(model, ids, trajectories, state_columns, battle_won, title="") -> plt.Figure`**
  — actual vs. simulated-from-start vs. simulated-from-midway, for a few episodes.
- **`save_run(runs_dir, name, config, model, state_columns, metrics, per_episode, figures) -> Path`**
  — writes `config.json` (+ `code_info()`), `model.json`, `equations.txt`, `metrics.json`,
  `per_episode.csv`, and the figures, into a new timestamped folder under `runs_dir`. Never
  overwrites (raises `FileExistsError` on a repeat folder name, same minute-resolution behavior as
  before).

## Notebook 04 changes

- Cells that defined `load_all_indexes`/`wilson_interval`/`summarize`/`heatmap`/`episodes_by_time`
  now just call the imported versions. Fixed a latent bug while doing this: the `heatmap()` call
  was passing a dead `fmt` argument positionally (`heatmap(summary, "win_rate", "{:.0%}", "Blue win
  rate")`) that the function never used — `evaluation/sweep.py`'s `heatmap()` dropped that
  parameter when it was extracted, so this call would have silently mis-bound `title`/`cmap` had
  it not been caught here.
- Cells that defined `to_trajectory`/`fit_sindy`/`simulate`/`evaluate`/`summarize_eval`/
  `plot_predictions`/`save_run` now import them and pass the explicit arguments (`STATE_COLUMNS`,
  `trajectories`, `dt_fit`, `SMOOTH_WINDOW`, `index["battle_won"]`) that used to be implicit
  globals.
- `run_config["dataset"]` now also records `"experiment": EXPERIMENT`, not just the dataset name
  — `exp01` and `exp02` reuse identical dataset names, so a saved run previously couldn't be
  traced back to which experiment produced it except by timestamp (this was a known gap noted in
  the 2026-10-08 session doc; fixed here as part of touching this cell anyway).

**Validated, not just refactored:** executed the whole notebook (every code cell except the final
`save_run`, which is covered by a unit test instead, to avoid writing a spurious extra run folder)
against the real `exp02` data on disk. Every printed number and every fitted equation came back
byte-for-byte identical to what the notebook had before the refactor — same win rates, same
`(health_blue)' = -0.076 1 + 0.080 health_blue + -0.169 health_red` equation, same `metrics` dict.
Confirms the move changed nothing about behavior, only where the code lives.

## New tests: `tests/test_sindy_models.py`

- `to_trajectory` trims to one step before first damage, and downsamples correctly.
- A synthetic linear-decay fixture runs `fit_sindy` → `simulate`/`evaluate`/`summarize_eval` →
  `save_run` end to end, checking shapes, that `save_run`'s files exist and contain what's
  expected, and that a repeat `save_run` call raises `FileExistsError` rather than overwriting.

## Also noticed while validating (not caused by this change)

Re-running notebook 05's comparison against the live data on disk found that **`exp01` now has
500 episodes per dataset (8000 total), not the 100 it had when the README's Findings numbers were
written** — someone re-collected it with a bigger `N_EPISODES` at some point, probably to match
`exp02`'s sample size for a fairer comparison (a sensible thing to do, just not reflected in the
docs yet). Recomputed on the current data, the numbers move a bit (e.g. `a090_s1`: was 89%, now
86%) but the overall pattern in the README's "From comparing exp01 vs exp02" section — the drop
concentrated at `step_mul` 1-2, the `aggression=1.0` fix, etc. — still holds. The README's two
sweep tables weren't updated here, since that's rewriting findings/interpretation rather than
moving code; flagged for a deliberate follow-up instead.
