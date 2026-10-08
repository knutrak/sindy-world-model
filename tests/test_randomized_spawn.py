"""
Does randomize_spawn actually move the units, and keep each episode reproducible?
"""

import pandas as pd

from sindy_wm.data import storage
from sindy_wm.data.collect import collect
from sindy_wm.envs.policies import AttackNearestPolicy


def test_randomize_spawn_varies_start_positions_and_is_reproducible(tmp_path):
    d1, d2 = tmp_path / "ds1", tmp_path / "ds2"
    kwargs = dict(
        policy=AttackNearestPolicy(aggression=0.9),
        description="test",
        step_mul=8,
        max_steps=20,
        n_episodes=3,
        allow_dirty=True,
        randomize_spawn=True,
        spawn_distance_range=(8.0, 16.0),
    )
    collect(dataset_dir=d1, **kwargs)
    collect(dataset_dir=d2, **kwargs)

    # Same seeds, randomize_spawn: episodes are still reproducible
    pd.testing.assert_frame_equal(storage.load_snapshots(d1), storage.load_snapshots(d2))

    # Different episodes (seeds) started from different positions
    snaps = storage.load_snapshots(d1)
    start = snaps[snaps["step"] == 0]
    blue0 = start[(start["team"] == "blue") & (start["unit_id"] == 0)]
    assert blue0.groupby("episode_idx")[["x", "y"]].first().nunique().min() > 1
