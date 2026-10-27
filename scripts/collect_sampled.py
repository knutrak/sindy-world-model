"""Collect exp05: the large sampled dataset, plus a held-out test set. Existing datasets are skipped.

Every battle draws its own Red unit count, force ratio, starting health and spawn
(EpisodeSampler), and one Blue policy setting that it keeps for the whole battle
(SwitchingPolicy with segment_seconds=None): each battle is one consistent system,
as plain SINDy (without control inputs) assumes. The setting is still logged per
step in controls/.

- main:     20,000 battles, seeds 0-19,999, for fitting and choosing models
- held-out:  5,000 battles, seeds from 1,000,000, the same settings; for final
             results only, never for choosing models

Thesis datasets: commit all changes first. With ALLOW_DIRTY = False collection
refuses to start on uncommitted changes, so the recorded commit describes the code
exactly. If a run is interrupted, delete the incomplete dataset folder (its
manifest says "in_progress") and run again; finished datasets are skipped.

Run from the project root (about 40 min and 1 GB in total):

    uv run scripts/collect_sampled.py
    uv run scripts/collect_sampled.py --scale 0.01 --experiment-dir /tmp/exp05 --allow-dirty   # quick test
"""

import argparse
from pathlib import Path

from sindy_wm.envs.episode_sampling import EpisodeSampler
from sindy_wm.envs.policies import PolicySpace, SwitchingPolicy
from sindy_wm.evaluation.coverage import print_coverage
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR

EXPERIMENT = "exp05_10m_sampled"
EXPERIMENT_DESCRIPTION = (
    "Large dataset for fitting team-level models: every battle draws its own Red unit count, "
    "force ratio, starting health, spawn and Blue policy setting (kept for the whole battle)."
)

SCENARIO = "10m_vs_10m"  # only its map and unit type are used: the sampler sets the unit counts
STEP_MUL = 2  # 0.125 s per step
MAX_SECONDS = 100  # battle time limit in game seconds, converted to steps
ALLOW_DIRTY = False  # thesis datasets: commit first

SAMPLER = EpisodeSampler(
    red_units=(6, 14),
    force_ratio=(0.75, 1.33),  # Blue units / Red units, drawn log-uniformly
    damaged_start_share=0.5,  # per side
    start_health=(0.4, 1.0),
    spawn_distance=(4.0, 16.0),  # below 5 (marine range) battles start in contact
)
POLICY = SwitchingPolicy(
    PolicySpace(unit_retreat_share=0.08, team_retreat_share=0.05),  # ~13% of battles retreat in some way
    segment_seconds=None,  # one setting per battle
)

DATASETS = [
    {
        "name": "10m_sampled_s2_v1",
        "first_seed": 0,
        "n_episodes": 20_000,
        "description": "Main set: for fitting and choosing models.",
    },
    {
        "name": "10m_sampled_s2_heldout_v1",
        "first_seed": 1_000_000,  # far from the main set's seeds: no battle appears in both
        "n_episodes": 5_000,
        "description": "Held-out test set: same settings, new seeds. For final results only.",
    },
]


def write_experiment_readme(experiment_dir: Path) -> None:
    """Write a short README for the experiment folder, unless one exists."""
    readme = experiment_dir / "README.md"
    if readme.exists():
        return
    experiment_dir.mkdir(parents=True, exist_ok=True)
    datasets = "\n".join(
        f"- {ds['name']}: {ds['n_episodes']} battles, seeds from {ds['first_seed']}. {ds['description']}"
        for ds in DATASETS
    )
    readme.write_text(
        f"# {EXPERIMENT}\n\n{EXPERIMENT_DESCRIPTION}\n\n"
        f"- Scenario map: {SCENARIO}, step_mul {STEP_MUL}, time limit {MAX_SECONDS} s\n"
        f"- Per-battle draws (episodes.jsonl): {SAMPLER}\n"
        f"- Blue policy (one setting per battle, logged in controls/): {POLICY.space}\n\n"
        f"{datasets}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiment-dir", type=Path, default=RAW_DIR / EXPERIMENT)
    parser.add_argument(
        "--scale", type=float, default=1.0, help="multiply every dataset's size, for test runs"
    )
    parser.add_argument("--allow-dirty", action="store_true", help="collect despite uncommitted changes")
    args = parser.parse_args()

    write_experiment_readme(args.experiment_dir)
    for ds in DATASETS:
        dataset_dir = args.experiment_dir / ds["name"]
        if (dataset_dir / "manifest.json").exists():
            print(f"Skipping {ds['name']}: already exists")
            continue
        collect(
            dataset_dir=dataset_dir,
            policy=POLICY,
            description=ds["description"],
            scenario=SCENARIO,
            step_mul=STEP_MUL,
            max_steps=MAX_SECONDS * 16 // STEP_MUL,
            n_episodes=max(1, round(ds["n_episodes"] * args.scale)),
            first_seed=ds["first_seed"],
            allow_dirty=ALLOW_DIRTY or args.allow_dirty,
            sampler=SAMPLER,
        )
        print_coverage(dataset_dir, max_episodes=1000)


if __name__ == "__main__":
    main()
