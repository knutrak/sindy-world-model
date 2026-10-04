"""Play one random episode of 3s5z and save per-unit snapshots."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import smaclite  # noqa: F401  (registers the environments)

from sindy_wm.envs.smaclite_snapshot import SnapshotLogger

SEED = 0
OUT = Path("data/raw/smoke_test_3s5z_seed0.parquet")


def main():
    env = gym.make("smaclite/3s5z-v0", use_cpp_rvo2=False)
    env.reset(seed=SEED)
    logger = SnapshotLogger(env)
    rng = np.random.default_rng(SEED)

    frames = [logger.start_episode()]
    done, step = False, 0
    while not done and step < 200:
        avail = env.unwrapped.get_avail_actions()
        actions = [int(rng.choice(np.flatnonzero(a))) for a in avail]
        _, _, done, truncated, _ = env.step(actions)
        done = done or truncated
        step += 1
        frames.append(logger.snapshot(step=step))

    df = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"Episode finished after {step} steps, saved {len(df)} rows to {OUT}")
    print(df.groupby(["step", "team"])["hp"].sum().unstack().tail())


if __name__ == "__main__":
    main()
