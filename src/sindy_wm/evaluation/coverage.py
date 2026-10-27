"""Coverage checks for datasets collected with per-episode sampling (`EpisodeSampler`).

Answers, for one dataset: are battles contested, does the data leave the
Blue = Red diagonal, does health still act as a clock for engagement, and how
are policy settings used? Used by scripts/collect_pilot.py and
scripts/collect_sampled.py after collecting.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sindy_wm.io import storage
from sindy_wm.states.aggregate import engagement

REFERENCE_UNITS = 10  # health is reported as "full marines / 10", the same scale for both sides
RATIO_BINS = [0, 0.8, 0.95, 1.05, 1.25, np.inf]


def print_coverage(dataset_dir: Path, max_episodes: int | None = 1000) -> None:
    """Print the coverage checks for a dataset collected with an EpisodeSampler and SwitchingPolicy.

    Outcomes use every episode. The checks that need snapshots (correlations,
    policy settings) use the first `max_episodes` (None: all), which keeps
    memory and time reasonable on large datasets.
    """
    manifest = storage.load_manifest(dataset_dir)
    dt = manifest["config"]["env"]["seconds_per_step"]
    index = storage.load_episode_index(dataset_dir).set_index("episode_idx")
    subset = list(index.index if max_episodes is None else index.index[:max_episodes])
    snaps = storage.load_snapshots(dataset_dir, subset)
    controls = storage.load_controls(dataset_dir, subset)
    size_mb = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()) / 1e6

    print(f"\n== {dataset_dir.name}: {len(index)} episodes, {size_mb:.0f} MB")
    duration = index["n_steps"] * dt
    print(
        f"Blue win rate {index['battle_won'].mean():.0%}, truncated {index['truncated'].mean():.1%}, "
        f"duration {duration.mean():.1f} s (max {duration.max():.1f} s)"
    )

    ratio = index["n_blue"] / index["n_red"]
    by_ratio = index.groupby(pd.cut(ratio, RATIO_BINS), observed=True)["battle_won"].agg(["size", "mean"])
    by_ratio.columns = ["episodes", "blue_win_rate"]
    print("\nBy force ratio (Blue / Red): a bin near 0% or 100% is decided from the start")
    print(by_ratio.round(2).to_string())
    damaged = (index[["start_health_blue", "start_health_red"]] < 1).sum(axis=1)
    print(
        f"Sides starting damaged: none {np.mean(damaged == 0):.0%}, one {np.mean(damaged == 1):.0%}, "
        f"both {np.mean(damaged == 2):.0%}"
    )

    # Team health on one common scale: full marines / REFERENCE_UNITS
    snaps["health"] = snaps["hp"] + snaps["shield"]
    marine_hp = snaps["max_hp"].iloc[0] + snaps["max_shield"].iloc[0]
    health = snaps.groupby(["episode_idx", "step", "team"])["health"].sum().unstack() / (
        marine_hp * REFERENCE_UNITS
    )
    print(f"\nOn {len(subset)} episodes:")
    print(
        f"corr(Blue health, Red health): {health['blue'].corr(health['red']):.2f}"
        "   (exp03, all battles starting at full health: 0.97; low = off the diagonal)"
    )

    # Does Red's health per unit still tell how far into the fight it is (share of its units in range)?
    eng = snaps.groupby("episode_idx").apply(engagement, include_groups=False)
    alive = snaps[snaps["alive"]].groupby(["episode_idx", "step", "team"]).size().unstack().reindex(eng.index)
    in_range = (eng["engaged_red"] / alive["red"]).where(alive["red"] > 0)
    started = eng["engaged_red"].groupby(level="episode_idx").cumsum() > 0
    per_unit = health["red"] * REFERENCE_UNITS / alive["red"]
    ok = in_range.notna() & started
    print(
        f"corr(Red health per unit, share of Red units in range): {per_unit[ok].corr(in_range[ok]):.2f}"
        "   (exp03: -0.72; closer to 0 = health is less of a clock)"
    )

    segments = controls.groupby("episode_idx")["segment"].max() + 1
    rules = controls["rule"].value_counts(normalize=True).round(2).to_dict()
    retreat_on = (controls["unit_below"] > 0) | (controls["team_below"] > 0)
    battles_retreating = retreat_on.groupby(controls["episode_idx"]).any().mean()
    span = f"min {segments.min()}, max {segments.max()}"
    print(f"\nPolicy settings per episode: mean {segments.mean():.1f} ({span})")
    print(f"Share of steps by target rule: {rules}")
    print(f"Retreat rule on in {battles_retreating:.0%} of battles, {retreat_on.mean():.0%} of steps")
