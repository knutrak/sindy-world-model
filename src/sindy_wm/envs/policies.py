"""Policies that choose actions for the Blue (allied) units.

A policy is used like this, once per episode:

    policy.reset(seed)                 # at the start of the episode
    actions = policy.act(env)          # every step: one action per Blue unit

Action indices follow SMAClite: 0 = no-op, 1 = stop, 2-5 = move
north/south/east/west, 6+ = attack enemy (index - 6).
"""

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np


class Policy(ABC):
    """Base class for all policies."""

    def reset(self, seed: int | None = None) -> None:
        """Prepare for a new episode. Seeds the policy's random generator."""
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, env: gym.Env) -> list[int]:
        """Return one action index per Blue unit for the current step."""


class RandomPolicy(Policy):
    """Each Blue unit picks uniformly among its currently available actions."""

    def act(self, env: gym.Env) -> list[int]:
        avail = env.unwrapped.get_avail_actions()
        return [int(self.rng.choice(np.flatnonzero(mask))) for mask in avail]


# Action indices in SMAClite
NOOP, STOP = 0, 1
MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST = 2, 3, 4, 5  # north = +y, east = +x
FIRST_TARGET = 6  # action FIRST_TARGET + k targets unit k of the other team


class AttackNearestPolicy(Policy):
    """Each Blue unit attacks the nearest target in range, otherwise moves closer.

    With probability `aggression` a unit follows this rule; otherwise it takes
    a random available action. Lower aggression gives more varied battles.
    """

    def __init__(self, aggression: float = 1.0):
        if not 0.0 <= aggression <= 1.0:
            raise ValueError(f"aggression must be between 0 and 1, got {aggression}")
        self.aggression = aggression

    def act(self, env: gym.Env) -> list[int]:
        game = env.unwrapped
        actions = []
        for agent_id, mask in enumerate(game.get_avail_actions()):
            unit = game.agents.get(agent_id)
            if unit is None:  # dead units can only no-op
                actions.append(NOOP)
            elif self.rng.random() < self.aggression:
                actions.append(self._attack_nearest(game, unit, mask))
            else:
                actions.append(int(self.rng.choice(np.flatnonzero(mask))))
        return actions

    def _attack_nearest(self, game, unit, mask: np.ndarray) -> int:
        # Enemies in range, according to the availability mask
        target_ids = np.flatnonzero(mask[FIRST_TARGET:])
        if len(target_ids) > 0:
            nearest = min(target_ids, key=lambda k: np.linalg.norm(game.enemies[k].pos - unit.pos))
            return FIRST_TARGET + int(nearest)

        # Nothing in range: move towards the nearest enemy
        if not game.enemies:
            return STOP
        enemy = min(game.enemies.values(), key=lambda e: np.linalg.norm(e.pos - unit.pos))
        dx, dy = enemy.pos - unit.pos
        horizontal = MOVE_EAST if dx > 0 else MOVE_WEST
        vertical = MOVE_NORTH if dy > 0 else MOVE_SOUTH
        # Prefer the axis with the larger gap; fall back to the other one
        preferred = [horizontal, vertical] if abs(dx) >= abs(dy) else [vertical, horizontal]
        for move in preferred:
            if mask[move]:
                return move
        return STOP
