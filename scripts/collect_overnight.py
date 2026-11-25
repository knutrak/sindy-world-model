"""Collect exp14: the blind test of docs/overnight/00_plan.md. Committed code only, seeds 40 x00 000 +.

uv run scripts/collect_overnight.py marines
"""

import argparse

from sindy_wm.envs.episode_sampling import CloseBattleSampler, MixedBattleSampler, TripleMixSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.io.collect import collect
from sindy_wm.paths import RAW_DIR

RAW = RAW_DIR / "exp14_blind"
SETS = {  # scenario, sampler, withdraw radius, first seed
    "marines": ("10m_vs_10m", CloseBattleSampler(), 7.0, 40_000_000),
    "marauders": ("10mrd_vs_10mrd", CloseBattleSampler(unit_size=1.125), 7 * 7.125 / 5.75, 40_100_000),
    "stalkers": ("10s_vs_10s", CloseBattleSampler(), 7 * 7.25 / 5.75, 40_200_000),
    "mixed": ("mixed_m_mrd", MixedBattleSampler(), 8.0, 40_300_000),
    "triple": ("mixed_m_mrd_s", TripleMixSampler(), 8.0, 40_400_000),
    "n1": ("10m_n1", CloseBattleSampler(), 7.0, 40_500_000),
    "n2": ("10m_n2", CloseBattleSampler(), 7.0, 40_600_000),
    "n3": ("10m_n3", CloseBattleSampler(), 7.0, 40_700_000),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=list(SETS))
    what = parser.parse_args().what
    scenario, sampler, wr, seed = SETS[what]
    collect(
        dataset_dir=RAW / what,
        policy=IntentSchedule(withdraw_radius=wr),
        description=f"exp14 {what}: blind test of the overnight experiment, withdraw radius {wr:.2f}.",
        scenario=scenario,
        step_mul=2,
        max_steps=90 * 16 // 2,
        n_episodes=1000,
        first_seed=seed,
        allow_dirty=False,
        sampler=sampler,
    )


if __name__ == "__main__":
    main()
