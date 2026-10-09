"""Saving and loading datasets of episodes.

A dataset is a folder with a manifest, an episode index, and one Parquet file
per episode for snapshots and for actions:

    data/raw/<dataset_name>/
        manifest.json           how the dataset was made (written once, finalized at the end)
        episodes.jsonl          one line per saved episode: seed, n_steps, outcome
        snapshots/
            episode_00000.parquet
            ...
        actions/
            episode_00000.parquet
            ...

Typical use, once per dataset:

    create_dataset(dataset_dir, config)       # before the first episode
    save_episode(dataset_dir, idx, episode)   # after every episode
    finalize_dataset(dataset_dir)             # when all episodes are done

Raw data is never overwritten: create_dataset refuses an existing dataset, and
save_episode refuses an episode that is already saved.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sindy_wm.provenance import code_info

if TYPE_CHECKING:
    from sindy_wm.envs.episode import Episode

SCHEMA_VERSION = 1

MANIFEST_FILE = "manifest.json"
EPISODES_FILE = "episodes.jsonl"
SNAPSHOT_DIR = "snapshots"
ACTION_DIR = "actions"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def episode_path(dataset_dir: str | Path, episode_idx: int, kind: str = SNAPSHOT_DIR) -> Path:
    """Return the file path for one episode's snapshots or actions.

    Args:
        dataset_dir: Folder of the dataset.
        episode_idx: Episode number, used in the file name.
        kind: "snapshots" or "actions".
    """
    if kind not in (SNAPSHOT_DIR, ACTION_DIR):
        raise ValueError(f"kind must be '{SNAPSHOT_DIR}' or '{ACTION_DIR}', got '{kind}'")
    return Path(dataset_dir) / kind / f"episode_{episode_idx:05d}.parquet"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def create_dataset(
    dataset_dir: str | Path,
    config: dict[str, Any],
    description: str = "",
    allow_dirty: bool = False,
) -> Path:
    """Create a new, empty dataset folder and write its manifest.

    Args:
        dataset_dir: Folder of the dataset, e.g. "data/raw/5m_attack_a08_v1".
            Must not exist yet, or be empty.
        config: Everything needed to regenerate the dataset, e.g.
            {"scenario": {...}, "env": {...}, "policy": {...}, "seeds": [...]}.
            Stored as given (must be JSON-serializable; numpy scalars are fine).
        description: Free text: why this dataset was collected.
        allow_dirty: By default, refuse to start if the git working tree has
            uncommitted changes, because the recorded commit would then not
            describe the code that produced the data. Set True to collect anyway;
            the manifest then records the dirty flag and the changed files.

    Returns:
        The path of the manifest.
    """
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        raise FileExistsError(f"{dataset_dir} already exists and is not empty. Use a new name.")

    code = code_info()
    if code["git_dirty"] and not allow_dirty:
        changed = "\n  ".join(code["git_changed_files"])
        raise RuntimeError(
            "Git working tree has uncommitted changes:\n  "
            f"{changed}\nCommit them first, or pass allow_dirty=True."
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": dataset_dir.name,
        "description": description,
        "status": "in_progress",
        "created": _now(),
        "finished": None,
        "n_episodes": None,
        "config": config,
        "code": code,
    }
    (dataset_dir / SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
    (dataset_dir / ACTION_DIR).mkdir(exist_ok=True)
    return _write_manifest(dataset_dir, manifest)


def save_episode(dataset_dir: str | Path, episode_idx: int, episode: Episode) -> dict[str, Any]:
    """Save one episode's snapshots and actions, and add it to the episode index.

    Both tables get an `episode_idx` column, so that the tables of many episodes
    can be loaded and concatenated in one go. The index line is written last,
    so the index only lists episodes whose files are completely saved.

    Returns:
        The record added to the episode index.
    """
    dataset_dir = Path(dataset_dir)
    if not (dataset_dir / MANIFEST_FILE).exists():
        raise FileNotFoundError(f"No manifest in {dataset_dir}. Call create_dataset first.")

    snap_path = episode_path(dataset_dir, episode_idx, SNAPSHOT_DIR)
    act_path = episode_path(dataset_dir, episode_idx, ACTION_DIR)
    for path in (snap_path, act_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists. Raw data is never overwritten.")

    _with_episode_idx(episode.snapshots, episode_idx).to_parquet(snap_path, index=False)
    _with_episode_idx(episode.actions, episode_idx).to_parquet(act_path, index=False)

    record = {
        "episode_idx": int(episode_idx),
        "seed": int(episode.seed),
        "n_steps": int(episode.n_steps),
        "battle_won": bool(episode.battle_won),
        "truncated": bool(episode.truncated),
        **_final_summary(episode.snapshots),
    }
    with open(dataset_dir / EPISODES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def finalize_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Mark a dataset as complete once all episodes are saved.

    Checks that every seed in config["seeds"] (if given) has exactly one saved
    episode, then records the finish time and episode count in the manifest.

    Returns:
        The updated manifest.
    """
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    index = load_episode_index(dataset_dir)

    if index["episode_idx"].duplicated().any():
        raise RuntimeError(f"Duplicate episode_idx in {EPISODES_FILE}.")

    seeds = manifest["config"].get("seeds")
    if seeds is not None:
        expected = _expected_seeds(seeds)
        saved = index["seed"].tolist()
        missing = sorted(set(expected) - set(saved))
        if missing or len(saved) != len(expected):
            raise RuntimeError(
                f"Saved {len(saved)} episodes, expected {len(expected)}. "
                f"Missing seeds (first 10): {missing[:10]}"
            )

    manifest["status"] = "complete"
    manifest["finished"] = _now()
    manifest["n_episodes"] = len(index)
    _write_manifest(dataset_dir, manifest)

    started = datetime.fromisoformat(manifest["created"])
    finished = datetime.fromisoformat(manifest["finished"])
    print(
        f"Finalized {manifest['name']}: {len(index)} episodes "
        f"in {finished - started}\n"
        f"  Blue win rate: {index['battle_won'].mean():.1%}   "
        f"truncated: {index['truncated'].mean():.1%}   "
        f"steps: mean {index['n_steps'].mean():.1f}, "
        f"min {index['n_steps'].min()}, max {index['n_steps'].max()}"
    )

    return manifest


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def load_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Load a dataset's manifest."""
    with open(Path(dataset_dir) / MANIFEST_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_episode_index(dataset_dir: str | Path) -> pd.DataFrame:
    """Load the episode index: one row per saved episode."""
    path = Path(dataset_dir) / EPISODES_FILE
    if not path.exists():
        return pd.DataFrame(columns=["episode_idx", "seed", "n_steps", "battle_won", "truncated"])
    return pd.read_json(path, lines=True).sort_values("episode_idx", ignore_index=True)


def load_snapshots(dataset_dir: str | Path, episode_idx: int | list[int] | None = None) -> pd.DataFrame:
    """Load snapshots of one episode, a list of episodes, or (None) all episodes."""
    return _load_tables(dataset_dir, SNAPSHOT_DIR, episode_idx)


def load_actions(dataset_dir: str | Path, episode_idx: int | list[int] | None = None) -> pd.DataFrame:
    """Load actions of one episode, a list of episodes, or (None) all episodes."""
    return _load_tables(dataset_dir, ACTION_DIR, episode_idx)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_episode_idx(table: pd.DataFrame, episode_idx: int) -> pd.DataFrame:
    return table.assign(episode_idx=episode_idx)[
        ["episode_idx", *[c for c in table.columns if c != "episode_idx"]]
    ]


def _expected_seeds(seeds: list[int] | dict[str, int]) -> list[int]:
    """Seeds from the manifest: a list, or {"first": a, "last": b} with b included."""
    if isinstance(seeds, dict):
        return list(range(seeds["first"], seeds["last"] + 1))
    return list(seeds)


def _final_summary(snapshots: pd.DataFrame) -> dict[str, Any]:
    """Alive units and total hp per team at the last step."""
    last = snapshots[snapshots["step"] == snapshots["step"].max()]
    summary = {}
    for team in ("blue", "red"):
        rows = last[last["team"] == team]
        summary[f"final_alive_{team}"] = int(rows["alive"].sum())
        summary[f"final_hp_{team}"] = float(rows["hp"].sum())
    return summary


def _load_tables(dataset_dir: str | Path, kind: str, episode_idx: int | list[int] | None) -> pd.DataFrame:
    if episode_idx is None:
        folder = Path(dataset_dir) / kind
        files = sorted(folder.glob("episode_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode files in {folder}")
    else:
        indices = [episode_idx] if isinstance(episode_idx, int) else list(episode_idx)
        files = [episode_path(dataset_dir, i, kind) for i in indices]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _write_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write the manifest atomically, so a crash never leaves a half-written file."""
    path = dataset_dir / MANIFEST_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _json_default(obj: Any) -> Any:
    """Convert numpy scalars and arrays, and paths, to plain JSON types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
