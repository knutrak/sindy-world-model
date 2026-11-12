"""Team-level intents for Blue: a small, discrete control input with tactical meaning.

An intent is a (movement, target) pair that the whole Blue team follows:

    movement   advance   attack what is in range, otherwise walk towards the enemy
               hold      hold the line: don't approach the enemy; once any Blue unit has an
                         enemy in range, every unit closes in and fights (as advance)
               withdraw  walk away from any enemy closer than `withdraw_radius`
                         (and so don't shoot); stand still once out of danger
    target     nearest   each unit shoots its nearest enemy in range
               focus     the team concentrates fire on one enemy (see `TargetChoicePolicy`)

`IntentPolicy` plays one intent. `IntentSchedule` switches between random intents
every few seconds (the data a model with control inputs needs) and can *branch*:
from a drawn moment on, it plays one given intent to the end. Playing the same
seed with different branch intents gives the same battle up to the branch and
one counterfactual future per intent.
"""

from itertools import product

import gymnasium as gym
import numpy as np

from sindy_wm.envs.policies import (
    FIRST_TARGET,
    NOOP,
    STOP,
    Policy,
    TargetChoicePolicy,
    _move_along,
    _nearest,
)

MOVEMENTS = ("advance", "hold", "withdraw")
TARGETS = ("nearest", "focus")
INTENTS = tuple(product(MOVEMENTS, TARGETS))  # all six (movement, target) pairs


class IntentPolicy(TargetChoicePolicy):
    """The Blue team follows one intent (see the module docstring). No random actions."""

    def __init__(self, movement: str = "advance", target: str = "nearest", withdraw_radius: float = 7.0):
        if movement not in MOVEMENTS:
            raise ValueError(f"movement must be one of {MOVEMENTS}, got {movement!r}")
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        super().__init__(rule=target, aggression=1.0)
        self.movement = movement
        self.target = target
        self.withdraw_radius = withdraw_radius

    def params(self) -> dict:
        return {"movement": self.movement, "target": self.target, "withdraw_radius": self.withdraw_radius}

    def act(self, env: gym.Env) -> list[int]:
        game = env.unwrapped
        masks = game.get_avail_actions()
        focus = self._focus_target(game, masks) if self.rule == "focus" else None
        # hold the line: the team is in contact once any alive Blue unit has a target in range
        self._contact = any(
            mask[FIRST_TARGET:].any() for agent_id, mask in enumerate(masks) if agent_id in game.agents
        )
        actions = []
        for agent_id, mask in enumerate(masks):
            unit = game.agents.get(agent_id)
            if unit is None:  # dead units can only no-op
                actions.append(NOOP)
            else:
                actions.append(self._intent_action(game, unit, mask, focus))
        return actions

    def _intent_action(self, game, unit, mask: np.ndarray, focus: int | None) -> int:
        if not game.enemies:
            return STOP
        if self.movement == "withdraw":
            enemy = _nearest(game.enemies.values(), unit.pos)
            away = unit.pos - enemy.pos
            if np.linalg.norm(away) <= self.withdraw_radius:
                move = _move_along(away, mask)
                if move != STOP:
                    return move
                # cornered: fight (falls through to the shooting rule)
        in_range = mask[FIRST_TARGET:].any()
        if in_range or self.movement == "advance" or (self.movement == "hold" and self._contact):
            return self._unit_action(game, unit, mask, focus)
        return STOP  # hold before contact, or withdrawn far enough: stand still


class IntentSchedule(Policy):
    """Random intents in segments of a few seconds, optionally branching into one fixed intent.

    Args:
        segment_seconds: (min, max) duration of each segment; None keeps the first
            intent for the whole episode.
        movement_weights: Probabilities of advance / hold / withdraw for each new segment.
        branch_intent: If given, this (movement, target) is played from the branch
            moment to the end of the episode.
        branch_seconds: (min, max) the branch moment is drawn from, per episode.
            (0, 0) branches at the start: a fixed intent for the whole episode.
        branch_hold_seconds: If given, the branch intent is played for this long and
            `then_intent` after that, to the end. None keeps the branch intent to the end.
        then_intent: The intent after the branch window (default: advance, nearest).

    Randomness: the segment schedule uses `default_rng([seed, 1])` and the branch
    moment `default_rng([seed, 3])`, so the same seed gives the same schedule and
    the same branch moment whatever `branch_intent` is.
    """

    def __init__(
        self,
        segment_seconds: tuple[float, float] | None = (1.0, 4.0),
        movement_weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
        branch_intent: tuple[str, str] | None = None,
        branch_seconds: tuple[float, float] = (0.0, 0.0),
        withdraw_radius: float = 7.0,
        branch_hold_seconds: float | None = None,
        then_intent: tuple[str, str] = ("advance", "nearest"),
    ):
        self.branch_hold_seconds = branch_hold_seconds
        self.then_intent = tuple(then_intent)
        self.segment_seconds = segment_seconds
        self.movement_weights = tuple(movement_weights)
        self.branch_intent = tuple(branch_intent) if branch_intent is not None else None
        self.branch_seconds = tuple(branch_seconds)
        self.withdraw_radius = withdraw_radius

    def params(self) -> dict:
        return {
            "segment_seconds": None if self.segment_seconds is None else list(self.segment_seconds),
            "movement_weights": list(self.movement_weights),
            "branch_intent": None if self.branch_intent is None else list(self.branch_intent),
            "branch_seconds": list(self.branch_seconds),
            "withdraw_radius": self.withdraw_radius,
            "branch_hold_seconds": self.branch_hold_seconds,
            "then_intent": list(self.then_intent),
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._schedule_rng = np.random.default_rng(None if seed is None else [seed, 1])
        branch_rng = np.random.default_rng(None if seed is None else [seed, 3])
        self._branch_at = float(branch_rng.uniform(*self.branch_seconds))  # drawn even if unused
        self._step = 0
        self._segment = -1
        self._next_switch = 0
        self._branched = False
        self._reverted = False
        self._current: IntentPolicy | None = None

    def act(self, env: gym.Env) -> list[int]:
        branch_step = round(self._branch_at / env.seconds_per_step)
        if self.branch_intent is not None and not self._branched and self._step >= branch_step:
            self._current = IntentPolicy(*self.branch_intent, withdraw_radius=self.withdraw_radius)
            self._branched = True
            self._segment += 1
        elif (
            self._branched
            and not self._reverted
            and self.branch_hold_seconds is not None
            and self._step >= branch_step + round(self.branch_hold_seconds / env.seconds_per_step)
        ):
            self._current = IntentPolicy(*self.then_intent, withdraw_radius=self.withdraw_radius)
            self._reverted = True
            self._segment += 1
        elif not self._branched and self._step >= self._next_switch:
            movement = MOVEMENTS[int(self._schedule_rng.choice(3, p=self.movement_weights))]
            target = TARGETS[int(self._schedule_rng.integers(2))]
            self._current = IntentPolicy(movement, target, withdraw_radius=self.withdraw_radius)
            if self.segment_seconds is None:
                self._next_switch = np.inf
            else:
                seconds = self._schedule_rng.uniform(*self.segment_seconds)
                self._next_switch = self._step + max(1, round(seconds / env.seconds_per_step))
            self._segment += 1
        self._step += 1
        return self._current.act(env)

    def setting(self) -> dict:
        if self._current is None:
            raise RuntimeError("No setting yet: call act() first.")
        return {
            "segment": self._segment,
            "movement": self._current.movement,
            "target": self._current.target,
            "branched": self._branched,
            "reverted": self._reverted,
            "branch_at": self._branch_at,
        }
