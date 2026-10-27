"""Collect a pilot of the sampled dataset, and check it before collecting the real thing.

Every episode draws its own Red unit count, force ratio, starting health and spawn
(EpisodeSampler), and Blue switches policy setting every 1-3 s (SwitchingPolicy).
After collecting (or if the pilot already exists), prints the checks that decide
whether the ranges are right:

- outcomes: win rate, overall and by force ratio, and how often battles hit the time limit
- coverage: is the data still stuck on the Blue = Red diagonal?
- the health "clock": does a side's health still tell how far into the battle it is?
- control: how many policy settings an episode gets

Run from the project root:

    uv run scripts/collect_pilot.py                    # 300 episodes into data/raw/
    uv run scripts/collect_pilot.py --n-episodes 50    # a quicker look
    uv run scripts/collect_pilot.py --experiment-dir /tmp/pilot   # somewhere else
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sindy_wm.envs.episode_sampling import EpisodeSampler
from sindy_wm.envs.policies import PolicySpace, SwitchingPolicy
from sindy_wm.io import storage
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR
from sindy_wm.states.aggregate import engagement

EXPERIMENT = "exp04_sampled_pilot"
DATASET = "10m_sampled_s2_pilot_v1"
DESCRIPTION = (
    "Pilot: per-episode Red count, force ratio, starting health and spawn; "
    "Blue switches RetreatPolicy settings every 1-3 s. Checks ranges before the large collection."
)

SCENARIO = "10m_vs_10m"  # only its map and unit type are used: the sampler sets the unit counts
STEP_MUL = 2  # 0.125 s per step
MAX_SECONDS = 100  # battle time limit in game seconds, converted to steps
N_EPISODES = 300
FIRST_SEED = 0
ALLOW_DIRTY = True  # fine for a pilot; set False (and commit first) for thesis datasets

SAMPLER = EpisodeSampler(
    red_units=(6, 14),
    force_ratio=(0.75, 1.33),  # Blue / Red, drawn log-uniformly: 0.75 and 1.33 mirror each other
    damaged_start_share=0.5,  # per side
    start_health=(0.4, 1.0),
    spawn_distance=(4.0, 16.0),  # below 5 (marine range) battles start in contact
)
POLICY = SwitchingPolicy(PolicySpace(), segment_seconds=(1.0, 3.0))

REFERENCE_UNITS = 10  # health is reported as "full marines / 10", the same scale for both sides
RATIO_BINS = [0, 0.8, 0.95, 1.05, 1.25, np.inf]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--experiment-dir", type=Path, default=RAW_DIR / EXPERIMENT)
    args = parser.parse_args()

    dataset_dir = args.experiment_dir / DATASET
    if (dataset_dir / "manifest.json").exists():
        print(f"{dataset_dir} already exists: summarizing it (delete it to collect again)\n")
    else:
        collect(
            dataset_dir=dataset_dir,
            policy=POLICY,
            description=DESCRIPTION,
            scenario=SCENARIO,
            step_mul=STEP_MUL,
            max_steps=MAX_SECONDS * 16 // STEP_MUL,
            n_episodes=args.n_episodes,
            first_seed=FIRST_SEED,
            allow_dirty=ALLOW_DIRTY,
            sampler=SAMPLER,
        )
    summarize(dataset_dir)


def summarize(dataset_dir: Path) -> None:
    """Print the pilot checks for a dataset collected with an EpisodeSampler and SwitchingPolicy."""
    manifest = storage.load_manifest(dataset_dir)
    dt = manifest["config"]["env"]["seconds_per_step"]
    index = storage.load_episode_index(dataset_dir).set_index("episode_idx")
    snaps = storage.load_snapshots(dataset_dir)
    controls = storage.load_controls(dataset_dir)
    size_mb = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()) / 1e6

    print(f"\n== {dataset_dir.name}: {len(index)} episodes, {size_mb:.0f} MB")
    duration = index["n_steps"] * dt
    print(
        f"Blue win rate {index['battle_won'].mean():.0%}, truncated {index['truncated'].mean():.0%}, "
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
    print(
        f"\ncorr(Blue health, Red health): {health['blue'].corr(health['red']):.2f}"
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
    retreating = ((controls["unit_below"] > 0) | (controls["team_below"] > 0)).mean()
    span = f"min {segments.min()}, max {segments.max()}"
    print(f"\nPolicy settings per episode: mean {segments.mean():.1f} ({span})")
    print(f"Share of steps by target rule: {rules}; steps with a retreat rule on: {retreating:.0%}")


if __name__ == "__main__":
    main()
