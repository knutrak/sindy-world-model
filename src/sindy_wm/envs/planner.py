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


class OraclePlannerPolicy(Policy):
    """The same planning scheme with the *real simulator* as the world model: an upper bound.

    Every `replan_seconds` it copies the game once per candidate, plays the candidate for
    `commit_seconds` and advance/nearest for the rest of `horizon_seconds` (or to the
    battle's end), and picks the best real health margin. What a perfect model would
    achieve with this planner.
    """

    def __init__(
        self,
        replan_seconds: float = 1.0,
        commit_seconds: float = 4.0,
        horizon_seconds: float = 12.0,
        candidates=CANDIDATES,
    ):
        self.replan_seconds, self.commit_seconds, self.horizon_seconds = (
            replan_seconds,
            commit_seconds,
            horizon_seconds,
        )
        self.candidates = tuple(candidates)

    def params(self) -> dict:
        return {
            "replan_seconds": self.replan_seconds,
            "commit_seconds": self.commit_seconds,
            "horizon_seconds": self.horizon_seconds,
            "candidates": [list(c) for c in self.candidates],
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._step = 0
        self._current = IntentPolicy(*self.candidates[0])

    @staticmethod
    def _margin(game) -> float:
        return sum(u.hp + u.shield for u in game.agents.values()) - sum(
            u.hp + u.shield for u in game.enemies.values()
        )

    def _lookahead(self, env: gym.Env, intent) -> float:
        import copy

        sim = copy.deepcopy(env)
        commit, steps = (
            round(self.commit_seconds / env.seconds_per_step),
            round(self.horizon_seconds / env.seconds_per_step),
        )
        first, then = IntentPolicy(*intent), IntentPolicy("advance", "nearest")
        for n in range(steps):
            _, _, terminated, truncated, _ = sim.step((first if n < commit else then).act(sim))
            if terminated or truncated:
                break
        return self._margin(sim.unwrapped)

    def act(self, env: gym.Env) -> list[int]:
        game = env.unwrapped
        if (
            self._step % max(1, round(self.replan_seconds / env.seconds_per_step)) == 0
            and game.agents
            and game.enemies
        ):
            scores = [self._lookahead(env, intent) for intent in self.candidates]
            self._current = IntentPolicy(*self.candidates[int(np.argmax(scores))])
        self._step += 1
        return self._current.act(env)

    def setting(self) -> dict:
        return {"movement": self._current.movement, "target": self._current.target}
