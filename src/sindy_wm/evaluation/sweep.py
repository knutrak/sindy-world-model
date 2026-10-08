"""Summarizing a sweep: one table across all datasets in an experiment, win rates, and heatmaps.

Used to analyse an experiment folder (many datasets varying step_mul/aggression/...)
and to compare experiments with each other (see notebooks 04 and 05).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sindy_wm.data import storage


def load_all_indexes(experiment_dir: str | Path) -> pd.DataFrame:
    """Episode indexes of all complete datasets in an experiment folder, plus their parameters.

    Skips anything in experiment_dir that isn't a complete dataset: the
    experiment's own README.md, and any dataset still `in_progress`.
    Durations are converted to seconds, since a step means something
    different at each step_mul.
    """
    experiment_dir = Path(experiment_dir)
    tables = []
    for d in sorted(experiment_dir.iterdir()):
        if not (d / "manifest.json").exists():
            continue
        m = storage.load_manifest(d)
        if m["status"] != "complete":
            print(f"Skipping {d.name}: status {m['status']}")
            continue
        cfg = m["config"]
        index = storage.load_episode_index(d)
        tables.append(
            index.assign(
                experiment=experiment_dir.name,
                dataset=d.name,
                step_mul=cfg["env"]["step_mul"],
                aggression=cfg["policy"]["params"].get("aggression"),
                policy=cfg["policy"]["class"],
                duration_s=index["n_steps"] * cfg["env"]["seconds_per_step"],
            )
        )
    if not tables:
        raise FileNotFoundError(f"No complete datasets in {experiment_dir}")
    return pd.concat(tables, ignore_index=True)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate (Wilson score interval, works near 0 and 1)."""
    if n == 0:
        return np.nan, np.nan
    p = wins / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def summarize(group: pd.DataFrame) -> pd.Series:
    """Win rate (with its 95% interval), duration, and outcome variety for one dataset's episodes."""
    wins, n = int(group["battle_won"].sum()), len(group)
    low, high = wilson_interval(wins, n)
    outcome_cols = [c for c in ["n_steps", "final_hp_blue", "final_hp_red"] if c in group]
    return pd.Series(
        {
            "n": n,
            "win_rate": wins / n,
            "win_low": low,
            "win_high": high,
            "duration_mean_s": group["duration_s"].mean(),
            "duration_std_s": group["duration_s"].std(),
            "steps_mean": group["n_steps"].mean(),
            "truncated": group["truncated"].mean(),
            "distinct_outcomes": len(group[outcome_cols].drop_duplicates()),
        }
    )


def heatmap(summary: pd.DataFrame, value: str, title: str, cmap: str = "RdBu", ax: plt.Axes | None = None):
    """Aggression x step_mul heatmap of one summary column, with a colorbar.

    Draws into `ax` if given (for side-by-side comparisons), otherwise creates
    its own figure. Returns the figure, axes, and the pivoted table (aggression
    x step_mul), so the caller can add its own cell annotations (see notebook
    04, Part 1.3).
    """
    table = summary.pivot(index="aggression", columns="step_mul", values=value)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    im = ax.imshow(
        table,
        cmap=cmap,
        vmin=0 if value == "win_rate" else None,
        vmax=1 if value == "win_rate" else None,
        aspect="auto",
    )
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("step_mul")
    ax.set_ylabel("aggression")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig, ax, table


def episodes_by_time(states: pd.DataFrame, episode_ids) -> list[pd.DataFrame]:
    """One table per episode, indexed by time in seconds (for plot_many_episodes)."""
    return [states.loc[i].set_index("t") for i in episode_ids]
