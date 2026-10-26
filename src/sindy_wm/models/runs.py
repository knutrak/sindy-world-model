"""Saving a fitted model and its evaluation as a run in runs/ (see the README's Runs section)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pysindy as ps

from sindy_wm.provenance import code_info


def save_run(
    runs_dir: Path,
    name: str,
    config: dict,
    model: ps.SINDy,
    state_columns: list[str],
    metrics: dict,
    per_episode: pd.DataFrame,
    figures: dict[str, plt.Figure],
) -> Path:
    """Save a run: config, model, equations, metrics, per-episode results, and figures.

    An existing run folder is never overwritten (see the README's Runs section).

    Args:
        runs_dir: Folder to create the new run folder in, e.g. `paths.PROJECT_ROOT / "runs"`.
        name: Short name included in the run folder's name.
        config: Dataset, split, and preprocessing/model choices (whatever isn't
            already implied by `model`); `code_info()` is added automatically.
        model: The fitted SINDy model.
        state_columns: Level A columns used as the SINDy state (recorded in model.json).
        metrics: Summary metrics on the test episodes, e.g. from `summarize_eval`.
        per_episode: Metrics per test episode, e.g. from `evaluate`.
        figures: filename -> figure, saved under the run's figures/ folder.

    Returns:
        The new run folder's path.
    """
    run_dir = Path(runs_dir) / f"{datetime.now():%Y-%m-%d_%H%M}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a run
    (run_dir / "figures").mkdir()

    def write_json(filename, obj):
        (run_dir / filename).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

    write_json("config.json", {**config, "code": code_info()})
    write_json(
        "model.json",
        {
            "state_names": state_columns,
            "feature_names": model.get_feature_names(),
            "coefficients": model.coefficients().tolist(),  # one row per state variable
            "equations": model.equations(precision=5),
        },
    )
    (run_dir / "equations.txt").write_text(
        "\n".join(f"({s})' = {eq}" for s, eq in zip(state_columns, model.equations(precision=5), strict=True))
        + "\n",
        encoding="utf-8",
    )
    write_json("metrics.json", metrics)
    per_episode.to_csv(run_dir / "per_episode.csv")
    for filename, fig in figures.items():
        fig.savefig(run_dir / "figures" / filename, dpi=150, bbox_inches="tight")
    return run_dir
