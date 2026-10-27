"""Policies that choose actions for the Blue (allied) units.

A policy is used like this, once per episode:

    policy.reset(seed)                 # at the start of the episode
    actions = policy.act(env)          # every step: one action per Blue unit

Action indices follow SMAClite: 0 = no-op, 1 = stop, 2-5 = move
north/south/east/west, 6+ = attack enemy (index - 6).
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np


class Policy(ABC):
    """Base class for all policies."""

    def config(self) -> dict:
        """Return a JSON-serializable dictionary describing the policy."""
        return {"class": type(self).__name__, "params": self.params()}

    def params(self) -> dict:
        """Parameters that change the policy's behaviour. Override if there are any."""
        return {}

    def setting(self) -> dict:
        """The setting in use right now, logged at every step by `play_episode`.

        For most policies that is just `params()`. A policy that changes its
        own setting during an episode (`SwitchingPolicy`) returns the current one.
        """
        return self.params()

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

    def params(self) -> dict:
        return {"aggression": self.aggression}

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
        enemy = _nearest(game.enemies.values(), unit.pos)
        return _move_along(enemy.pos - unit.pos, mask)


TARGET_RULES = ("nearest", "weakest", "focus")


class TargetChoicePolicy(Policy):
    """Each Blue unit attacks an enemy in range chosen by `rule`, otherwise moves closer.

    Rules, among the enemies a unit can shoot right now:

    - "nearest": the closest one. Identical to `AttackNearestPolicy`.
    - "weakest": the one with the least health left (hp + shield), to finish
      off damaged enemies and remove their firepower sooner.
    - "focus": the whole team picks one target, the weakest enemy that any
      Blue unit can shoot, and every unit that can reach it shoots it. Units
      that can't, shoot their own weakest target in range instead.

    With nothing in range, a unit moves towards the nearest enemy (the team's
    focus target, for "focus"). With probability `1 - aggression` a unit takes
    a random available action instead, as in `AttackNearestPolicy`.

    The rule decides how damage is spread over the enemy team, and so how
    quickly enemy units die (alive) for the same health lost: "focus" kills
    units soonest, "nearest" spreads damage the most.
    """

    def __init__(self, rule: str = "nearest", aggression: float = 1.0):
        if rule not in TARGET_RULES:
            raise ValueError(f"rule must be one of {TARGET_RULES}, got {rule!r}")
        if not 0.0 <= aggression <= 1.0:
            raise ValueError(f"aggression must be between 0 and 1, got {aggression}")
        self.rule = rule
        self.aggression = aggression

    def params(self) -> dict:
        return {"rule": self.rule, "aggression": self.aggression}

    def act(self, env: gym.Env) -> list[int]:
        game = env.unwrapped
        masks = game.get_avail_actions()
        focus = self._focus_target(game, masks) if self.rule == "focus" else None
        actions = []
        for agent_id, mask in enumerate(masks):
            unit = game.agents.get(agent_id)
            if unit is None:  # dead units can only no-op
                actions.append(NOOP)
            elif self.rng.random() < self.aggression:
                actions.append(self._unit_action(game, unit, mask, focus))
            else:
                actions.append(int(self.rng.choice(np.flatnonzero(mask))))
        return actions

    def _unit_action(self, game, unit, mask: np.ndarray, focus: int | None) -> int:
        """The rule's action for one unit that follows it this step."""
        target_ids = np.flatnonzero(mask[FIRST_TARGET:])
        if len(target_ids) > 0:
            if self.rule == "nearest":
                k = min(target_ids, key=lambda k: np.linalg.norm(game.enemies[k].pos - unit.pos))
            elif focus is not None and focus in target_ids:
                k = focus
            else:  # "weakest", or "focus" with the focus target out of this unit's reach
                k = min(target_ids, key=lambda k: (_health(game.enemies[k]), k))
            return FIRST_TARGET + int(k)

        # Nothing in range: move towards the nearest enemy (or the focus target)
        if not game.enemies:
            return STOP
        enemy = game.enemies[focus] if focus is not None else _nearest(game.enemies.values(), unit.pos)
        return _move_along(enemy.pos - unit.pos, mask)

    @staticmethod
    def _focus_target(game, masks) -> int | None:
        """The team's target: the weakest enemy any Blue unit can shoot, else the one nearest Blue."""
        in_reach = {
            int(k)
            for agent_id, mask in enumerate(masks)
            if agent_id in game.agents
            for k in np.flatnonzero(mask[FIRST_TARGET:])
        }
        if in_reach:
            return min(in_reach, key=lambda k: (_health(game.enemies[k]), k))
        if not game.enemies or not game.agents:
            return None
        centre = np.mean([u.pos for u in game.agents.values()], axis=0)
        return min(game.enemies, key=lambda k: np.linalg.norm(game.enemies[k].pos - centre))


class RetreatPolicy(TargetChoicePolicy):
    """`TargetChoicePolicy`, but units pull back from nearby enemies when weak.

    A Blue unit retreats (moves directly away from the nearest enemy) when an
    enemy is within `danger_radius` and either

    - its own health (hp + shield) is below `unit_below` of its maximum, or
    - Blue's total health is below `team_below` times Red's total health
      (the whole team falls back when it is losing).

    Otherwise it follows the target rule. A retreating unit doesn't shoot, so
    retreat trades firepower now for units (and their firepower) later. With
    the defaults (0 and 0) it never retreats and equals `TargetChoicePolicy`.
    Units do not heal, so a weak unit keeps retreating whenever enemies come
    close; out of danger it rejoins the fight, so it tends to hover at the edge.

    `danger_radius` should exceed the enemies' attack range (5 for marines),
    or a retreating unit is already being shot when it starts to retreat.
    Random actions (`1 - aggression`) apply to retreating units too.
    """

    def __init__(
        self,
        rule: str = "nearest",
        aggression: float = 1.0,
        unit_below: float = 0.0,
        team_below: float = 0.0,
        danger_radius: float = 6.0,
    ):
        super().__init__(rule, aggression)
        if not 0.0 <= unit_below <= 1.0:
            raise ValueError(f"unit_below must be between 0 and 1, got {unit_below}")
        if team_below < 0.0:
            raise ValueError(f"team_below must be at least 0, got {team_below}")
        self.unit_below = unit_below
        self.team_below = team_below
        self.danger_radius = danger_radius

    def params(self) -> dict:
        return {
            **super().params(),
            "unit_below": self.unit_below,
            "team_below": self.team_below,
            "danger_radius": self.danger_radius,
        }

    def act(self, env: gym.Env) -> list[int]:
        game = env.unwrapped
        masks = game.get_avail_actions()
        focus = self._focus_target(game, masks) if self.rule == "focus" else None
        blue = sum(_health(u) for u in game.agents.values())
        red = sum(_health(u) for u in game.enemies.values())
        team_retreats = red > 0 and blue < self.team_below * red
        actions = []
        for agent_id, mask in enumerate(masks):
            unit = game.agents.get(agent_id)
            if unit is None:  # dead units can only no-op
                actions.append(NOOP)
            elif self.rng.random() < self.aggression:
                retreat = self._retreat(game, unit, mask, team_retreats)
                actions.append(retreat if retreat is not None else self._unit_action(game, unit, mask, focus))
            else:
                actions.append(int(self.rng.choice(np.flatnonzero(mask))))
        return actions

    def _retreat(self, game, unit, mask: np.ndarray, team_retreats: bool) -> int | None:
        """A move away from the nearest enemy if this unit should retreat now, else None."""
        if not game.enemies:
            return None
        weak = _health(unit) < self.unit_below * (unit.max_hp + unit.max_shield)
        if not (weak or team_retreats):
            return None
        enemy = _nearest(game.enemies.values(), unit.pos)
        away = unit.pos - enemy.pos
        if np.linalg.norm(away) > self.danger_radius:
            return None
        move = _move_along(away, mask)
        return move if move != STOP else None  # cornered: fight instead


@dataclass(frozen=True)
class PolicySpace:
    """The settings a `SwitchingPolicy` draws from: a random `RetreatPolicy` each time.

    Every draw picks a target rule and an aggression; with probability
    `unit_retreat_share` weak units retreat (below a drawn `unit_below`), and
    with probability `team_retreat_share` the team retreats when losing (below
    a drawn `team_below`). Otherwise those thresholds are 0 (no retreat).
    """

    rules: tuple[str, ...] = TARGET_RULES
    aggression: tuple[float, float] = (0.7, 1.0)
    unit_retreat_share: float = 0.3
    unit_below: tuple[float, float] = (0.2, 0.5)
    team_retreat_share: float = 0.2
    team_below: tuple[float, float] = (0.5, 0.9)
    danger_radius: float = 6.0

    def sample(self, rng: np.random.Generator) -> RetreatPolicy:
        # Every value is drawn every time, so one draw always uses the same
        # number of random numbers, whichever branches are taken
        rule = self.rules[int(rng.integers(len(self.rules)))]
        aggression, unit_below, team_below = (
            rng.uniform(*r) for r in (self.aggression, self.unit_below, self.team_below)
        )
        unit_retreats, team_retreats = rng.random(2) < (self.unit_retreat_share, self.team_retreat_share)
        return RetreatPolicy(
            rule,
            float(aggression),
            unit_below=float(unit_below) if unit_retreats else 0.0,
            team_below=float(team_below) if team_retreats else 0.0,
            danger_radius=self.danger_radius,
        )


class SwitchingPolicy(Policy):
    """Switches to a new random setting from `space` every few seconds of a battle.

    Each segment of the episode uses one `RetreatPolicy` drawn from `space`,
    for a duration drawn from `segment_seconds`. So actions change *within*
    an episode, which a model with control inputs needs (a setting fixed for
    the whole episode is just a constant). `setting()` returns the current
    segment's index and parameters; `play_episode` logs it at every step.

    Randomness: the units' actions use `default_rng(seed)`, as every policy;
    the schedule (which settings, how long) uses its own stream,
    `default_rng([seed, 1])`, so changing the schedule never changes how the
    random actions within a segment are drawn.

    Needs the environment from `make_env`, whose `seconds_per_step` converts
    segment durations to steps.
    """

    def __init__(self, space: PolicySpace | None = None, segment_seconds: tuple[float, float] = (1.0, 3.0)):
        if not 0 < segment_seconds[0] <= segment_seconds[1]:
            raise ValueError(f"segment_seconds must be (min, max) with 0 < min <= max, got {segment_seconds}")
        self.space = space or PolicySpace()
        self.segment_seconds = segment_seconds

    def params(self) -> dict:
        return {"space": asdict(self.space), "segment_seconds": list(self.segment_seconds)}

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._schedule_rng = np.random.default_rng(None if seed is None else [seed, 1])
        self._step = 0
        self._segment = -1
        self._next_switch = 0
        self._current: RetreatPolicy | None = None

    def act(self, env: gym.Env) -> list[int]:
        if self._step >= self._next_switch:
            self._current = self.space.sample(self._schedule_rng)
            self._current.rng = self.rng  # the units' random actions keep using the episode's stream
            seconds = self._schedule_rng.uniform(*self.segment_seconds)
            self._next_switch = self._step + max(1, round(seconds / env.seconds_per_step))
            self._segment += 1
        self._step += 1
        return self._current.act(env)

    def setting(self) -> dict:
        if self._current is None:
            raise RuntimeError("No setting yet: call act() first.")
        return {"segment": self._segment, **self._current.params()}


def _health(unit) -> float:
    return float(unit.hp + unit.shield)


def _nearest(units, pos: np.ndarray):
    return min(units, key=lambda u: np.linalg.norm(u.pos - pos))


def _move_along(direction: np.ndarray, mask: np.ndarray) -> int:
    """The available move closest to `direction`: the axis with the larger component first, then the other."""
    dx, dy = direction
    horizontal = MOVE_EAST if dx > 0 else MOVE_WEST
    vertical = MOVE_NORTH if dy > 0 else MOVE_SOUTH
    preferred = [horizontal, vertical] if abs(dx) >= abs(dy) else [vertical, horizontal]
    for move in preferred:
        if mask[move]:
            return move
    return STOP
