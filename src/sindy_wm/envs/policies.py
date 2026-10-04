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