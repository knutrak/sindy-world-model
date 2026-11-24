"""Team-level intents for Blue: a small, discrete control input with tactical meaning.

An intent is a (movement, target) pair that the whole Blue team follows:

    movement   advance   attack what is in range, otherwise walk towards the enemy
               hold      hold the line: don't approach the enemy; once any Blue unit has an
                         enemy in range, every unit closes in and fights (as advance)
               withdraw  walk away from any enemy closer than `withdraw_radius`
                         (and so don't shoot); stand still once out of danger
               kite      (22 Nov, docs/planning/26) shoot when the weapon is ready, step away from the
                         nearest enemy while it reloads, advance with nothing in range
               stand     hold position (21 Nov, docs/planning/17): never move; shoot only enemies
                         inside the unit's own attack range (an attack order on a target that is
                         offered but out of range would make the unit walk), else stand still.
                         Not in MOVEMENTS / INTENTS, so the existing intent sets are unchanged
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
EXTRA_MOVEMENTS = ("stand", "kite", "spread", "flank")  # accepted by IntentPolicy, not part of INTENTS
SPREAD_R = 2.2  # docs/counters/00: the baneling splash radius
FLANK_W, FLANK_ARRIVE, FLANK_MAX_S = 7.0, 3.5, 10.0  # docs/counters/02: waypoint offset, arrival, timeout
FLANK_STALL = (1.0, 0.3)  # docs/counters/02 amendment 2: attack if the half got < 0.3 closer in the last 1 s
TARGETS = ("nearest", "focus")
INTENTS = tuple(product(MOVEMENTS, TARGETS))  # all six (movement, target) pairs


class IntentPolicy(TargetChoicePolicy):
    """The Blue team follows one intent (see the module docstring). No random actions."""

    def __init__(
        self,
        movement: str = "advance",
        target: str = "nearest",
        withdraw_radius: float = 7.0,
        bounds: tuple[float, float, float, float] | None = None,
    ):
        if movement not in MOVEMENTS + EXTRA_MOVEMENTS:
            raise ValueError(f"movement must be one of {MOVEMENTS + EXTRA_MOVEMENTS}, got {movement!r}")
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        super().__init__(rule=target, aggression=1.0)
        self.movement = movement
        self.target = target
        self.withdraw_radius = withdraw_radius
        self.bounds = bounds  # (x0, x1, y0, y1), used by flank to keep its waypoints on the map
        self._flank = None
        self._step = 0

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._flank = None
        self._step = 0

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
        if self.movement == "flank":
            self._flank_update(env, game)
        self._step += 1
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
        if self.movement == "stand":
            return self._stand_action(game, unit, mask, focus)
        if self.movement == "kite":
            return self._kite_action(game, unit, mask, focus)
        if self.movement == "spread":
            return self._spread_action(game, unit, mask, focus)
        if self.movement == "flank":
            return self._flank_action(game, unit, mask, focus)
        in_range = mask[FIRST_TARGET:].any()
        if in_range or self.movement == "advance" or (self.movement == "hold" and self._contact):
            return self._unit_action(game, unit, mask, focus)
        return STOP  # hold before contact, or withdrawn far enough: stand still

    def _stand_action(self, game, unit, mask: np.ndarray, focus: int | None) -> int:
        """Hold position: attack an enemy inside the unit's own attack range (by the target rule), else
        STOP."""
        offered = np.flatnonzero(mask[FIRST_TARGET:])
        reach = [int(k) for k in offered if unit.has_within_attack_range(game.enemies[int(k)])]
        if not reach:
            return STOP
        if self.rule == "focus" and focus in reach:
            k = focus
        elif self.rule == "focus":
            k = min(reach, key=lambda k: (game.enemies[k].hp + game.enemies[k].shield, k))
        else:
            k = min(reach, key=lambda k: np.linalg.norm(game.enemies[k].pos - unit.pos))
        return FIRST_TARGET + k

    def _spread_action(self, game, unit, mask: np.ndarray, focus: int | None) -> int:
        """Spread (docs/counters/00): as advance, but a unit that cannot fire this step steps away from a
        friend closer than SPREAD_R (the baneling splash radius)."""
        offered = np.flatnonzero(mask[FIRST_TARGET:])
        can_fire = unit.cooldown <= KITE_READY and any(
            unit.has_within_attack_range(game.enemies[int(k)]) for k in offered
        )
        if not can_fire:
            friends = [u for u in game.agents.values() if u is not unit]
            if friends:
                f = _nearest(friends, unit.pos)
                if np.linalg.norm(f.pos - unit.pos) < SPREAD_R:
                    move = _move_along(unit.pos - f.pos, mask)
                    if move != STOP:
                        return move
        return self._unit_action(game, unit, mask, focus)

    def _flank_update(self, env, game) -> None:
        """Flank (docs/counters/02): at the first step, split the team into two halves across the line to the
        enemy and fix a waypoint beside the enemy for each; a half attacks once it arrives or after
        FLANK_MAX_S."""
        if self._flank is None:
            ids = list(game.agents)
            pos = np.array([game.agents[i].pos for i in ids])
            c = pos.mean(0)
            e = np.mean([u.pos for u in game.enemies.values()], axis=0)
            a = (e - c) / max(np.linalg.norm(e - c), 1e-9)
            perp = np.array([-a[1], a[0]])
            order = np.argsort((pos - c) @ perp)
            half = len(ids) // 2
            groups = {-1: [ids[k] for k in order[:half]], 1: [ids[k] for k in order[half:]]}
            way = {}
            for side in (-1, 1):
                w = e + side * FLANK_W * perp
                if self.bounds is not None:
                    x0, x1, y0, y1 = self.bounds
                    w = np.array([np.clip(w[0], x0 + 1, x1 - 1), np.clip(w[1], y0 + 1, y1 - 1)])
                way[side] = w
            self._flank = {
                "side": {i: s for s, g in groups.items() for i in g},
                "way": way,
                "groups": groups,
                "attack": {-1: not groups[-1], 1: not groups[1]},
                "hist": {-1: [], 1: []},
            }
        f = self._flank
        late = self._step * env.seconds_per_step >= FLANK_MAX_S
        lag = max(1, round(FLANK_STALL[0] / env.seconds_per_step))
        for side, g in f["groups"].items():
            alive = [game.agents[i].pos for i in g if i in game.agents]
            if f["attack"][side]:
                continue
            if late or not alive:
                f["attack"][side] = True
                continue
            d = float(np.linalg.norm(np.mean(alive, axis=0) - f["way"][side]))
            h = f["hist"][side]
            h.append(d)
            if d <= FLANK_ARRIVE or (len(h) > lag and h[-1 - lag] - d < FLANK_STALL[1]):
                f["attack"][side] = True

    def _flank_action(self, game, unit, mask: np.ndarray, focus: int | None) -> int:
        """Flank, per unit: walk to the half's waypoint (fight only an enemy inside the unit's own range);
        advance once the half attacks."""
        f = self._flank
        side = f["side"][unit.id_in_faction]
        offered = np.flatnonzero(mask[FIRST_TARGET:])
        if f["attack"][side] or any(unit.has_within_attack_range(game.enemies[int(k)]) for k in offered):
            return self._unit_action(game, unit, mask, focus)
        move = _move_along(f["way"][side] - unit.pos, mask)
        return move if move != STOP else self._unit_action(game, unit, mask, focus)

    def _kite_action(self, game, unit, mask: np.ndarray, focus: int | None) -> int:
        """Kite (docs/planning/26): with an enemy inside the unit's own range, shoot when the weapon is ready
        (by the target rule) and step away from the nearest enemy while it reloads; otherwise advance."""
        offered = np.flatnonzero(mask[FIRST_TARGET:])
        reach = [int(k) for k in offered if unit.has_within_attack_range(game.enemies[int(k)])]
        if not reach:
            return self._unit_action(game, unit, mask, focus)
        if unit.cooldown > KITE_READY:
            enemy = _nearest(game.enemies.values(), unit.pos)
            move = _move_along(unit.pos - enemy.pos, mask)
            if move != STOP:
                return move
        return self._stand_action(game, unit, mask, focus)


KITE_READY = 0.125  # seconds of cooldown left at which a kiting unit turns to fire (one step)


class IntentSchedule(Policy):
    """Random intents in segments of a few seconds, optionally branching into one fixed intent.

    Args:
        segment_seconds: (min, max) duration of each segment; None keeps the first
            intent for the whole episode.
        movement_weights: Probabilities of advance / hold / withdraw for each new segment; a fourth weight
            adds kite (docs/planning/32). Three weights draw exactly as before.
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
        movement_weights: tuple[float, ...] = (0.5, 0.3, 0.2),
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
            names = MOVEMENTS if len(self.movement_weights) == 3 else (*MOVEMENTS, "kite")
            movement = names[int(self._schedule_rng.choice(len(names), p=self.movement_weights))]
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
