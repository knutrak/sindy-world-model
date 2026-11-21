"""A planner for Blue that chooses intents by simulating a world model.

Every `replan_seconds` it reads the team state from the live game, rolls the model
forward once per candidate intent (the intent for `commit_seconds`, then
advance/nearest to the end of `horizon_seconds`), and plays the intent with the
best predicted health margin (Blue minus Red) at the horizon.
"""

import gymnasium as gym
import numpy as np

from sindy_wm.envs.intents import INTENTS, IntentPolicy
from sindy_wm.envs.policies import Policy
from sindy_wm.models.intent_model import DT, SHAPE, STATE, control_vector
from sindy_wm.states.geometry import state_from_game

# withdraw ignores the target rule, so five candidates are distinct
CANDIDATES = tuple(i for i in INTENTS if i != ("withdraw", "focus"))


class ModelPlannerPolicy(Policy):
    """Chooses among `candidates` with a fitted `IntentModel` (see the module docstring).

    Args:
        model: A fitted `models.intent_model.IntentModel`.
        replan_seconds: How often a new intent is chosen.
        commit_seconds: How long the candidate intent is assumed to be played in the rollout.
        horizon_seconds: Length of the rollout; the score is the health margin at its end.
        shape: "model" or "hold", passed to `IntentModel.simulate`.
        use_reach: Start the rollout from the measured reach (relaxing to the closure).
        switch_margin: A new intent must beat the current one by this much to replace it.
    """

    def __init__(
        self,
        model,
        replan_seconds: float = 1.0,
        commit_seconds: float = 4.0,
        horizon_seconds: float = 12.0,
        shape: str = "model",
        use_reach: bool = True,
        switch_margin: float = 0.0,
        candidates=CANDIDATES,
    ):
        self.model = model
        self.replan_seconds, self.commit_seconds, self.horizon_seconds = (
            replan_seconds,
            commit_seconds,
            horizon_seconds,
        )
        self.shape, self.use_reach, self.switch_margin = shape, use_reach, switch_margin
        self.candidates = tuple(candidates)
        steps, commit = round(horizon_seconds / DT), round(commit_seconds / DT)
        default = control_vector("advance", "nearest")
        self._U = np.stack([np.repeat(default[None], steps, 0) for _ in self.candidates])
        for j, intent in enumerate(self.candidates):
            self._U[j, :commit] = control_vector(*intent)

    def params(self) -> dict:
        return {
            "replan_seconds": self.replan_seconds,
            "commit_seconds": self.commit_seconds,
            "horizon_seconds": self.horizon_seconds,
            "shape": self.shape,
            "use_reach": self.use_reach,
            "switch_margin": self.switch_margin,
            "candidates": [list(c) for c in self.candidates],
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._step = 0
        self._choice = 0
        self._current = IntentPolicy(*self.candidates[0])
        self.scores = None

    def predict(self, game) -> np.ndarray:
        """Predicted health margin at the horizon for each candidate, from the game's current state."""
        s = state_from_game(game)
        n = len(self.candidates)
        X0 = np.repeat(np.array([[s[c] for c in STATE]]), n, 0)
        shape0 = np.repeat(np.array([[s[c] for c in SHAPE]]), n, 0)
        reach0 = np.repeat(np.array([[s["reach_blue"], s["reach_red"]]]), n, 0) if self.use_reach else None
        Y = self.model.simulate(X0, shape0, self._U, reach0=reach0, shape=self.shape)
        return Y[:, -1, 0] - Y[:, -1, 1]

    def act(self, env: gym.Env) -> list[int]:
        game = env.unwrapped
        if (
            self._step % max(1, round(self.replan_seconds / env.seconds_per_step)) == 0
            and game.agents
            and game.enemies
        ):
            self.scores = self.predict(game)
            best = int(np.argmax(self.scores))
            if self.scores[best] > self.scores[self._choice] + self.switch_margin or self._step == 0:
                self._choice = best
                self._current = IntentPolicy(*self.candidates[best])
        self._step += 1
        return self._current.act(env)

    def setting(self) -> dict:
        return {"movement": self._current.movement, "target": self._current.target}


def typed_batch_from_row(row, L, U: np.ndarray) -> dict:
    """A `Pipeline.rollout` batch: one per-type state row (`states.typed.typed_states` columns), repeated
    once per control sequence in U (n, steps, 4)."""
    n = len(U)

    def rep(cols, fill=None):
        v = np.array([[float(row[c]) for c in cols]], float)
        if fill is not None:
            v = np.nan_to_num(v, nan=fill)
        return np.repeat(v, n, 0)

    return {
        "X0": rep(L.x_cols),
        "shape0": rep(SHAPE, 0.0),
        "front0": rep(L.front_cols, 0.0),
        "reach0": rep(L.r_cols),
        "U": U,
    }


class PipelinePlannerPolicy(Policy):
    """Chooses among `candidates` with a frozen `models.pipeline.Pipeline` (docs/planning/07_phase_b_plan.md).

    Every `replan_seconds` it reads the per-type state from the live game (one snapshot -> `typed_states`,
    reach capped at the model's `max_reach`), rolls the model forward once per candidate (the candidate for
    `commit_seconds`, then advance/nearest to `horizon_seconds`), and plays the candidate with the best
    predicted health margin (Blue − Red, team view) at the horizon. `log` keeps the scores of every re-plan.
    """

    def __init__(
        self,
        model,
        replan_seconds: float = 1.0,
        commit_seconds: float = 2.0,
        horizon_seconds: float = 3.0,
        withdraw_radius: float = 7.0,
        candidates=CANDIDATES,
    ):
        self.model = model
        self.replan_seconds, self.commit_seconds, self.horizon_seconds = (
            replan_seconds,
            commit_seconds,
            horizon_seconds,
        )
        self.withdraw_radius = withdraw_radius
        self.candidates = tuple(candidates)
        self.types = tuple(t.upper() for t in model.L.types)
        steps, commit = round(horizon_seconds / DT), round(commit_seconds / DT)
        default = control_vector("advance", "nearest")
        self._U = np.stack([np.repeat(default[None], steps, 0) for _ in self.candidates])
        for j, intent in enumerate(self.candidates):
            self._U[j, :commit] = control_vector(*intent)

    def params(self) -> dict:
        return {
            "replan_seconds": self.replan_seconds,
            "commit_seconds": self.commit_seconds,
            "horizon_seconds": self.horizon_seconds,
            "withdraw_radius": self.withdraw_radius,
            "candidates": [list(c) for c in self.candidates],
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._step = 0
        self._logger = None
        self._current = IntentPolicy(*self.candidates[0], withdraw_radius=self.withdraw_radius)
        self.log = []

    def live_state(self, env):
        """The per-type state row of the live game (the roster is recorded on the first call, at step 0)."""
        from sindy_wm.envs.smaclite_snapshot import SnapshotLogger
        from sindy_wm.states.typed import typed_states

        if self._logger is None:
            self._logger = SnapshotLogger(env)
            snap = self._logger.start_episode()
        else:
            snap = self._logger.snapshot(step=0)
        return typed_states(snap, self.types, max_reach=self.model.max_reach).iloc[0]

    def predict(self, env) -> np.ndarray:
        """Predicted health margin at the horizon for each candidate, from the live game's state."""
        B = typed_batch_from_row(self.live_state(env), self.model.L, self._U)
        with np.errstate(invalid="ignore", divide="ignore"):
            Y = self.model.L.team_view(self.model.rollout(B))
        return Y[:, -1, 0] - Y[:, -1, 1]

    def act(self, env: gym.Env) -> list[int]:
        game = env.unwrapped
        if self._logger is None:
            self.live_state(env)  # record the roster while every unit is alive
        if (
            self._step % max(1, round(self.replan_seconds / env.seconds_per_step)) == 0
            and game.agents
            and game.enemies
        ):
            scores = self.predict(env)
            best = int(np.argmax(scores))
            self.log.append(scores)
            self._current = IntentPolicy(*self.candidates[best], withdraw_radius=self.withdraw_radius)
        self._step += 1
        return self._current.act(env)

    def setting(self) -> dict:
        return {"movement": self._current.movement, "target": self._current.target}


class OraclePlannerPolicy(Policy):
    """The same planning scheme with the *real simulator* as the world model: an upper bound.

    Every `replan_seconds` it copies the game once per candidate, plays the candidate for
    `commit_seconds` and advance/nearest for the rest of `horizon_seconds` (or to the
    battle's end), and picks the best real health margin. What a perfect model would
    achieve with this planner.

    Options added for the headroom test (docs/planning/00_headroom_plan.md); the defaults keep the
    original behaviour:
        withdraw_radius: passed to the intents (7.0 for marines, 8.0 in mixed battles).
        perturb: (position, health) noise applied to each copy before its look-ahead: every unit's
            position moves by up to +/- position in x and y, its hp by up to +/- health (a fraction).
            Separates choosing well from knowing the future exactly.
        objective: "margin" (Blue minus Red health, hp + shield) or "survival" (Blue units alive,
            the margin as a tie-breaker).
        reseed: give each copy its own stream for SMAClite's per-tick unit-order shuffle (its only
            in-battle randomness), so the copies do not share the real game's future order.
    """

    def __init__(
        self,
        replan_seconds: float = 1.0,
        commit_seconds: float = 4.0,
        horizon_seconds: float = 12.0,
        candidates=CANDIDATES,
        withdraw_radius: float = 7.0,
        perturb: tuple[float, float] | None = None,
        objective: str = "margin",
        reseed: bool = False,
    ):
        self.replan_seconds, self.commit_seconds, self.horizon_seconds = (
            replan_seconds,
            commit_seconds,
            horizon_seconds,
        )
        self.candidates = tuple(candidates)
        if objective not in ("margin", "survival"):
            raise ValueError(objective)
        self.withdraw_radius, self.perturb, self.objective = withdraw_radius, perturb, objective
        self.reseed = reseed

    def params(self) -> dict:
        return {
            "replan_seconds": self.replan_seconds,
            "commit_seconds": self.commit_seconds,
            "horizon_seconds": self.horizon_seconds,
            "candidates": [list(c) for c in self.candidates],
            "withdraw_radius": self.withdraw_radius,
            "perturb": None if self.perturb is None else list(self.perturb),
            "objective": self.objective,
            "reseed": self.reseed,
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._step = 0
        self._current = IntentPolicy(*self.candidates[0], withdraw_radius=self.withdraw_radius)
        self._noise = np.random.default_rng(None if seed is None else [seed, 7])

    def _score(self, game) -> float:
        if self.objective == "survival":
            return len(game.agents) + 1e-4 * self._margin(game)
        return self._margin(game)

    def _perturb(self, game) -> None:
        dpos, dhp = self.perturb
        for unit in [*game.agents.values(), *game.enemies.values()]:
            unit.pos = unit.pos + self._noise.uniform(-dpos, dpos, 2)
            unit.hp = float(np.clip(unit.hp * (1 + self._noise.uniform(-dhp, dhp)), 1.0, unit.max_hp))

    @staticmethod
    def _margin(game) -> float:
        return sum(u.hp + u.shield for u in game.agents.values()) - sum(
            u.hp + u.shield for u in game.enemies.values()
        )

    def _lookahead(self, env: gym.Env, intent) -> float:
        import copy

        sim = copy.deepcopy(env)
        if self.perturb is not None:
            self._perturb(sim.unwrapped)
        if self.reseed:
            sim.unwrapped._np_random = np.random.default_rng(self._noise.integers(2**63))
        commit, steps = (
            round(self.commit_seconds / env.seconds_per_step),
            round(self.horizon_seconds / env.seconds_per_step),
        )
        first = IntentPolicy(*intent, withdraw_radius=self.withdraw_radius)
        then = IntentPolicy("advance", "nearest", withdraw_radius=self.withdraw_radius)
        for n in range(steps):
            _, _, terminated, truncated, _ = sim.step((first if n < commit else then).act(sim))
            if terminated or truncated:
                break
        return self._score(sim.unwrapped)

    def act(self, env: gym.Env) -> list[int]:
        game = env.unwrapped
        if (
            self._step % max(1, round(self.replan_seconds / env.seconds_per_step)) == 0
            and game.agents
            and game.enemies
        ):
            scores = [self._lookahead(env, intent) for intent in self.candidates]
            best = self.candidates[int(np.argmax(scores))]
            self._current = IntentPolicy(*best, withdraw_radius=self.withdraw_radius)
        self._step += 1
        return self._current.act(env)

    def setting(self) -> dict:
        return {"movement": self._current.movement, "target": self._current.target}
