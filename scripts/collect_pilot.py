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

from sindy_wm.envs.episode_sampling import EpisodeSampler
from sindy_wm.envs.policies import PolicySpace, SwitchingPolicy
from sindy_wm.evaluation.coverage import print_coverage
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR

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
POLICY = SwitchingPolicy(
    PolicySpace(unit_retreat_share=0.08, team_retreat_share=0.05),  # ~13% of settings retreat in some way
    segment_seconds=(1.0, 3.0),
)


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
    print_coverage(dataset_dir, max_episodes=None)


if __name__ == "__main__":
    main()
