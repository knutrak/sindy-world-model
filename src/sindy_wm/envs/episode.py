"""Playing complete episodes in SMAClite and recording what happens."""

from dataclasses import dataclass

import gymnasium as gym
import pandas as pd

from sindy_wm.envs.policies import Policy
from sindy_wm.envs.smaclite_snapshot import SnapshotLogger


@dataclass
class Episode:
    """Everything recorded from one episode.

    Attributes:
        snapshots: Per-unit table, one snapshot per step (steps 0..n_steps).
        actions: Blue actions, one row per unit per step. The action at step t
            is chosen from the snapshot at step t and leads to step t + 1.
        seed: Seed used for both the environment and the policy.
        n_steps: Number of env.step calls made.
        battle_won: True if all Red units were killed.
        truncated: True if the episode was stopped by max_steps.
    """

    snapshots: pd.DataFrame
    actions: pd.DataFrame
    seed: int
    n_steps: int
    battle_won: bool
    truncated: bool


def play_episode(env: gym.Env, policy: Policy, seed: int, max_steps: int = 200) -> Episode:
    """Play one episode with the given policy and record it.

    SMAClite has no time limit of its own, so max_steps stops episodes that
    would otherwise run forever (e.g. if Blue keeps retreating).
    """
    env.reset(seed=seed)
    policy.reset(seed)
    logger = SnapshotLogger(env)
    frames = [logger.start_episode()]
    # SMAClite expects actions ordered by Blue unit id 0, 1, ..., n_agents - 1
    blue_ids = sorted(uid for team, uid in logger.roster if team == "blue")

    action_rows = []
    battle_won = terminated = truncated = False
    step = 0

    while not (terminated or truncated):
        if step >= max_steps:
            truncated = True
            break
        actions = policy.act(env)
        action_rows += [
            {"step": step, "unit_id": uid, "action": a} for uid, a in zip(blue_ids, actions, strict=True)
        ]
        _, _, terminated, env_truncated, info = env.step(actions)
        step += 1
        frames.append(logger.snapshot(step=step))
        battle_won = info.get("battle_won", False)
        truncated = env_truncated

    return Episode(
        snapshots=pd.concat(frames, ignore_index=True),
        actions=pd.DataFrame(action_rows, columns=["step", "unit_id", "action"]),
        seed=seed,
        n_steps=step,
        battle_won=battle_won,
        truncated=truncated,
    )
