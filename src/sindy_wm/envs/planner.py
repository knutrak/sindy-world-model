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


def lanchester_value(Z: np.ndarray, k_blue: float, k_red: float) -> np.ndarray:
    """Square-law projection of the final margin from a team-view state Z (..., [H_B, H_R, A_B, A_R, d]):
    strengths S = k A H; the stronger side keeps H sqrt(1 - S_weak / S_strong) (docs/planning/25)."""
    hb, hr, ab, ar = Z[..., 0], Z[..., 1], Z[..., 2], Z[..., 3]
    sb, sr = k_blue * ab * hb, k_red * ar * hr
    with np.errstate(invalid="ignore", divide="ignore"):
        blue = hb * np.sqrt(np.clip(1 - np.divide(sr, sb, out=np.ones_like(sb), where=sb > 0), 0, 1))
        red = hr * np.sqrt(np.clip(1 - np.divide(sb, sr, out=np.ones_like(sr), where=sr > 0), 0, 1))
    return blue - red


def team_firepower(model) -> tuple[float, float]:
    """k_blue, k_red: the mean reach coefficient in the other side's health equations (docs/planning/25)."""
    out = {}
    for side, target in (("blue", "red"), ("red", "blue")):
        vals = [
            abs(v)
            for (s, _), c in model.coef.items()
            if s == target
            for k, v in c.items()
            if v and k.startswith("R[")
        ]
        out[side] = float(np.mean(vals)) if vals else 0.0
    return out["blue"], out["red"]


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
        vec = getattr(model, "control_vector", control_vector)  # the kite variant has five controls
        default = vec("advance", "nearest")
        self._U = np.stack([np.repeat(default[None], steps, 0) for _ in self.candidates])
        for j, intent in enumerate(self.candidates):
            self._U[j, :commit] = vec(*intent)

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
            self._snap0 = snap  # start health per unit, for the compartment states (docs/planning/13)
        else:
            snap = self._logger.snapshot(step=0)
        self._snap = snap
        return typed_states(snap, self.types, max_reach=self.model.max_reach).iloc[0]

    def live_spread(self) -> np.ndarray:
        """The spread state (docs/planning/09) of the last live snapshot, in the reach / front order."""
        from sindy_wm.states.spread import spread

        sp = spread(self._snap.assign(episode_idx=0), self.types).iloc[0]
        even = self.model.spread_model["measure"] == "even"
        cols = [
            f"{'even' if even else 'fragile'}_{side}_{t}"
            for side in ("blue", "red")
            for t in self.model.L.types
        ]
        v = sp[cols].to_numpy(float)
        return v if even else 1 - v

    def live_compartments(self) -> dict:
        """W0, HW0, h00 (docs/planning/13) of the last live snapshot, in the reach / front order."""
        from sindy_wm.states.compartments import compartments, unit_health

        s0 = self._snap0.assign(episode_idx=0)
        start = s0.assign(h=unit_health(s0)).set_index(["episode_idx", "team", "unit_id"])["h"]
        c = compartments(self._snap.assign(episode_idx=0), self.types, start=start).iloc[0]
        keys = [f"{side}_{t}" for side in ("blue", "red") for t in self.model.L.types]
        return {
            k0: c[[f"{k}_{x}" for x in keys]].to_numpy(float)
            for k0, k in (("W0", "W"), ("HW0", "HW"), ("h00", "h0"))
        }

    def live_margins(self) -> dict:
        """mu0, sd0: mean and spread of the reach margins (states/margins.py) of the last live snapshot,
        in the reach / front order (docs/planning/34)."""
        from sindy_wm.states.margins import margins

        r = margins(self._snap, self.types).iloc[0]
        keys = [f"{side}_{t}" for side in ("blue", "red") for t in self.model.L.types]
        return {k0: r[[f"{k}_{x}" for x in keys]].to_numpy(float) for k0, k in (("mu0", "mu"), ("sd0", "sd"))}

    def predict(self, env) -> np.ndarray:
        """Predicted health margin at the horizon for each candidate, from the live game's state."""
        B = typed_batch_from_row(self.live_state(env), self.model.L, self._U)
        if getattr(self.model, "spread_model", None) is not None:
            B["spread0"] = np.repeat(self.live_spread()[None], len(self._U), 0)
        if getattr(self.model, "compart_model", None) is not None:
            for k, v in self.live_compartments().items():
                B[k] = np.repeat(v[None], len(self._U), 0)
        # μ, σ reach states (docs/planning/34), or a model that takes them as inputs (docs/planning/36)
        if getattr(self.model, "ms_names", None) is not None or getattr(self.model, "needs_margins", False):
            for k, v in self.live_margins().items():
                B[k] = np.repeat(v[None], len(self._U), 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Y = self.model.L.team_view(self.model.rollout(B))
        if (
            getattr(self.model, "clipped", None) is not None
            and getattr(self.model, "bounds", None) is not None
        ):
            self.clip_log = getattr(self, "clip_log", [])
            self.clip_log.append(float(self.model.clipped.mean()))
        if getattr(self, "terminal", None) is not None:  # docs/planning/25: projected final margin
            return lanchester_value(Y[:, -1], *self.terminal)
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


class GatedPlannerPolicy(PipelinePlannerPolicy):
    """`PipelinePlannerPolicy` that plays a default intent unless the model predicts another candidate
    beats it by
    more than `tau` (docs/planning/22_gated_planner_plan.md). `deviations` counts the re-plans off the
    default."""

    def __init__(self, model, default=("hold", "focus"), tau: float = 0.0, **kw):
        super().__init__(model, **kw)
        self.default, self.tau = tuple(default), float(tau)
        self._d = self.candidates.index(self.default)

    def params(self) -> dict:
        return super().params() | {"default": list(self.default), "tau": self.tau}

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.deviations = 0
        self._current = IntentPolicy(*self.default, withdraw_radius=self.withdraw_radius)

    def act(self, env: gym.Env) -> list[int]:
        game = env.unwrapped
        if self._logger is None:
            self.live_state(env)
        if (
            self._step % max(1, round(self.replan_seconds / env.seconds_per_step)) == 0
            and game.agents
            and game.enemies
        ):
            scores = self.predict(env)
            self.log.append(scores)
            best = int(np.argmax(scores))
            pick = best if scores[best] - scores[self._d] > self.tau else self._d
            self.deviations += int(pick != self._d)
            self._current = IntentPolicy(*self.candidates[pick], withdraw_radius=self.withdraw_radius)
        self._step += 1
        return self._current.act(env)


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
        shared_noise: bool = False,
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
        # one perturbation draw per re-plan, applied to every candidate's copy (as a model's error is shared
        # by the candidates it compares); by default each copy gets its own draw
        self.shared_noise = shared_noise

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
            "shared_noise": self.shared_noise,
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._step = 0
        self._current = IntentPolicy(*self.candidates[0], withdraw_radius=self.withdraw_radius)
        self._noise = np.random.default_rng(None if seed is None else [seed, 7])
        self.score_log, self._draw = [], None

    def _score(self, game) -> float:
        if self.objective == "survival":
            return len(game.agents) + 1e-4 * self._margin(game)
        return self._margin(game)

    def _perturb(self, game) -> None:
        dpos, dhp = self.perturb
        draw = self._draw if self.shared_noise else None
        units = [
            *(("a", k, u) for k, u in game.agents.items()),
            *(("e", k, u) for k, u in game.enemies.items()),
        ]
        for side, k, unit in units:
            if draw is not None:
                shift, scale = draw[(side, k)]
            else:
                shift, scale = self._noise.uniform(-dpos, dpos, 2), self._noise.uniform(-dhp, dhp)
            unit.pos = unit.pos + shift
            unit.hp = float(np.clip(unit.hp * (1 + scale), 1.0, unit.max_hp))

    def _new_draw(self, game) -> None:
        """One shared perturbation per unit for this re-plan (shared_noise)."""
        dpos, dhp = self.perturb
        keys = [*(("a", k) for k in game.agents), *(("e", k) for k in game.enemies)]
        self._draw = {k: (self._noise.uniform(-dpos, dpos, 2), self._noise.uniform(-dhp, dhp)) for k in keys}

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
            if self.perturb is not None and self.shared_noise:
                self._new_draw(game)
            scores = [self._lookahead(env, intent) for intent in self.candidates]
            self.score_log.append(scores)
            best = self.candidates[int(np.argmax(scores))]
            self._current = IntentPolicy(*best, withdraw_radius=self.withdraw_radius)
        self._step += 1
        return self._current.act(env)

    def setting(self) -> dict:
        return {"movement": self._current.movement, "target": self._current.target}
