"""
Does saved data come back unchanged?
"""

from dataclasses import dataclass

import pandas as pd
import pytest

from sindy_wm.data import storage


@dataclass
class FakeEpisode:
    snapshots: pd.DataFrame
    actions: pd.DataFrame
    seed: int = 0
    n_steps: int = 2
    battle_won: bool = True
    truncated: bool = False


def fake_episode(seed=0):
    snaps = pd.DataFrame(
        {
            "step": [0, 0, 1, 1],
            "team": ["blue", "red"] * 2,
            "unit_id": [0, 0, 0, 0],
            "hp": [45.0, 45.0, 45.0, 0.0],
            "alive": [True, True, True, False],
        }
    )
    acts = pd.DataFrame({"step": [0, 1], "unit_id": [0, 0], "action": [6, 6]})
    return FakeEpisode(snaps, acts, seed=seed)


def test_round_trip(tmp_path):
    d = tmp_path / "ds"
    storage.create_dataset(d, {"seeds": [0]}, allow_dirty=True)
    ep = fake_episode()
    storage.save_episode(d, 0, ep)

    loaded = storage.load_snapshots(d, 0).drop(columns="episode_idx")
    pd.testing.assert_frame_equal(loaded, ep.snapshots)


def test_never_overwrites(tmp_path):
    d = tmp_path / "ds"
    storage.create_dataset(d, {"seeds": [0]}, allow_dirty=True)
    storage.save_episode(d, 0, fake_episode())
    with pytest.raises(FileExistsError):
        storage.save_episode(d, 0, fake_episode())


def test_finalize_detects_missing_seed(tmp_path):
    d = tmp_path / "ds"
    storage.create_dataset(d, {"seeds": [0, 1]}, allow_dirty=True)
    storage.save_episode(d, 0, fake_episode(seed=0))
    with pytest.raises(RuntimeError):
        storage.finalize_dataset(d)
