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
        controls: The policy's setting at every step (`Policy.setting()`), one
            row per step: the control input a model with control would use.
    """

    snapshots: pd.DataFrame
    actions: pd.DataFrame
    seed: int
    n_steps: int
    battle_won: bool
    truncated: bool
    controls: pd.DataFrame | None = None


def play_episode(
    env: gym.Env,
    policy: Policy,
    seed: int,
    max_steps: int = 200,
    start_health: tuple[float, float] | None = None,
) -> Episode:
    """Play one episode with the given policy and record it.

    SMAClite has no time limit of its own, so max_steps stops episodes that
    would otherwise run forever (e.g. if Blue keeps retreating).

    Args:
        env: Environment, from `make_env`.
        policy: Policy choosing Blue's actions.
        seed: Seed for both the environment and the policy.
        max_steps: Episodes are truncated after this many env.step calls.
        start_health: (Blue, Red) fraction of full health (hp and shield) that
            every unit of that side starts with, e.g. (1.0, 0.6) for a damaged
            Red side. Applied right after reset, so the step-0 snapshot shows it.
    """
    env.reset(seed=seed)
    if start_health is not None:
        _set_start_health(env, *start_health)
    policy.reset(seed)
    logger = SnapshotLogger(env)
    frames = [logger.start_episode()]
    # SMAClite expects actions ordered by Blue unit id 0, 1, ..., n_agents - 1
    blue_ids = sorted(uid for team, uid in logger.roster if team == "blue")

    action_rows, control_rows = [], []
    battle_won = terminated = truncated = False
    step = 0

    while not (terminated or truncated):
        if step >= max_steps:
            truncated = True
            break
        actions = policy.act(env)
        control_rows.append({"step": step, **policy.setting()})
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
        controls=pd.DataFrame(control_rows),
    )


def _set_start_health(env: gym.Env, blue: float, red: float) -> None:
    """Scale every unit's hp and shield to a fraction of its maximum (at least 1 hp, so it is alive)."""
    for fraction in (blue, red):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"start_health fractions must be in (0, 1], got {fraction}")
    game = env.unwrapped
    for units, fraction in ((game.agents, blue), (game.enemies, red)):
        for unit in units.values():
            unit.hp = max(1.0, fraction * unit.max_hp)
            unit.shield = fraction * unit.max_shield
