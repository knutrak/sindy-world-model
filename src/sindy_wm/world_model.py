"""The world-model interface: load a frozen SINDy model, describe a battle, ask what happens next.

    from sindy_wm.world_model import WorldModel

    wm = WorldModel.load("marines")                 # or "marauders", "mixed"
    state = wm.observe(env)                         # from a live SMAClite game, or
    state = wm.state_from_units(units)              # from any table of units (team, unit_type, x, y, hp)
    pred = wm.predict(state, "hold/focus", horizon=2.0)
    print(pred.to_text())                           # readable by a person or an LLM
    wm.compare(state)                               # every intent, best predicted margin first

Everything that goes in or comes out is in game units: hp, number of units, map units, seconds. The
equations and coefficients are those of the frozen pipeline (`models.pipeline`, docs/generalisation/06);
this module only wraps them, and `tests/test_world_model.py` checks that the numbers are identical.

How far to trust it: `docs/world_model.md`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sindy_wm.models.intent_model import DT, SHAPE, control_vector
from sindy_wm.models.pipeline import SIDES, Pipeline
from sindy_wm.paths import PROJECT_ROOT
from sindy_wm.states.geometry import REF_UNITS
from sindy_wm.states.typed import typed_states, unit_stats

MODEL_DIR = PROJECT_ROOT / "configs" / "world_models"
MOVEMENTS = ("advance", "hold", "withdraw")
TARGETS = ("nearest", "focus")
# withdraw ignores the target rule, so five intents are distinct (as in the planner experiments)
INTENTS = ("advance/nearest", "advance/focus", "hold/nearest", "hold/focus", "withdraw/nearest")
FORMAT = 1
STAND_OFF = (
    "the model predicts a stand-off (no damage, little movement); on training battles the real one fought on "
    "in 10-35% of such cases"
)
_SPEC = ("withdraw", "alloc", "thr_health", "thr_distance", "max_reach", "movement_wd", "withdraw_radius")

Intent = str | tuple[str, str]
Plan = Intent | Sequence[tuple[Intent, float]]


def parse_intent(intent: Intent) -> tuple[str, str]:
    """(movement, target) from "hold/focus" or ("hold", "focus"); a movement alone gets target "nearest"."""
    if not isinstance(intent, str | tuple | list) or not all(isinstance(p, str) for p in intent):
        raise ValueError(f"unknown intent {intent!r}: use '<movement>/<target>'")
    parts = tuple(intent.split("/")) if isinstance(intent, str) else tuple(intent)
    if len(parts) == 1:
        parts = (parts[0], "nearest")
    if len(parts) != 2 or parts[0] not in MOVEMENTS or parts[1] not in TARGETS:
        raise ValueError(
            f"unknown intent {intent!r}: use <movement>/<target> with movement in {MOVEMENTS} and target in "
            f"{TARGETS} (the model was fitted on these only)"
        )
    return parts


def _is_intent(plan) -> bool:
    """One intent, or a sequence of (intent, seconds)? Anything else is refused."""
    if isinstance(plan, str):
        return True
    if not isinstance(plan, list | tuple):
        raise ValueError(f"a plan is an intent or a list of (intent, seconds), got {plan!r}")
    if len(plan) == 2 and all(isinstance(p, str) for p in plan):
        return True
    for step in plan:
        if not isinstance(step, list | tuple) or len(step) != 2 or isinstance(step[1], str):
            raise ValueError(f"a plan is an intent or a list of (intent, seconds), got {plan!r}")
    return False


def plan_text(plan: Plan) -> str:
    if _is_intent(plan):
        return "/".join(parse_intent(plan))
    return ", then ".join(f"{'/'.join(parse_intent(i))} for {s:g} s" for i, s in plan)


def _steps(seconds: float) -> int:
    """Seconds as model steps, rounded to the nearest 0.125 s."""
    if isinstance(seconds, str) or not np.isfinite(seconds):
        raise ValueError(f"durations must be finite numbers of seconds, got {seconds!r}")
    n = round(seconds / DT)
    if n < 1:
        raise ValueError(f"{seconds} s is shorter than one model step ({DT} s)")
    return n


def plan_controls(plan: Plan, horizon: float | None) -> np.ndarray:
    """(steps, 4) control sequence. Without `horizon` a list of (intent, seconds) plays to its end; with it,
    the plan is cut, or its last intent is continued."""
    if _is_intent(plan):
        if horizon is None:
            raise ValueError("a single intent needs a horizon")
        return np.repeat(control_vector(*parse_intent(plan))[None], _steps(horizon), 0)
    if not len(plan):
        raise ValueError("empty plan")
    U = np.concatenate(
        [np.repeat(control_vector(*parse_intent(i))[None], _steps(seconds), 0) for i, seconds in plan]
    )
    if horizon is None:
        return U
    n = _steps(horizon)
    return np.concatenate([U, np.repeat(U[-1:], max(0, n - len(U)), 0)])[:n]


@dataclass(frozen=True)
class State:
    """One moment of a battle as the model sees it. Make it with `WorldModel.observe` / `state_from_units`."""

    row: pd.Series  # `states.typed.typed_states` columns, in the model's internal scales
    types: tuple[str, ...]
    hp_per_unit: dict[str, float]

    def hp(self, side: str) -> float:
        return float(
            sum(self.row[f"health_{side}_{t}"] * REF_UNITS * self.hp_per_unit[t] for t in self.types)
        )

    def units(self, side: str, unit_type: str | None = None) -> float:
        types = self.types if unit_type is None else (unit_type.lower(),)
        return float(sum(self.row[f"alive_{side}_{t}"] for t in types) * REF_UNITS)

    def in_reach(self, side: str) -> float:
        """Units of `side` with at least one enemy inside their reach."""
        return float(sum(self.row[f"reach_{side}_{t}"] for t in self.types) * REF_UNITS)

    @property
    def distance(self) -> float:
        return float(self.row["distance"])

    def to_dict(self) -> dict:
        out = {"distance": self.distance}
        for s in SIDES:
            out[s] = {"hp": self.hp(s), "in_reach": self.in_reach(s)}
            out[s]["units"] = {t: self.units(s, t) for t in self.types}
        return out

    def to_text(self) -> str:
        lines = [f"Distance between the team centres: {self.distance:.1f} map units."]
        for s in SIDES:
            mix = ", ".join(f"{self.units(s, t):.0f} {t}" for t in self.types if self.units(s, t))
            reach = f"{self.in_reach(s):.0f} with an enemy in reach"
            lines.append(f"{s.capitalize()}: {mix}; {self.hp(s):.0f} hp; {reach}.")
        return "\n".join(lines)


def _stand_off(trajectory: pd.DataFrame) -> bool:
    """Almost no damage and little movement (real closing is 3-6 per second) in the last second of a rollout
    of at least 2 s. The closing gate can stop shallow formations just outside reach (docs/world_model.md)."""
    if len(trajectory) <= round(2.0 / DT):
        return False
    last, end = trajectory.iloc[-1 - round(1.0 / DT)], trajectory.iloc[-1]
    hp = trajectory.iloc[0]["hp_blue"] + trajectory.iloc[0]["hp_red"]
    lost = sum(last[f"hp_{s}"] - end[f"hp_{s}"] for s in SIDES)
    return bool(lost < 0.01 * hp and abs(end["distance"] - last["distance"]) < 0.5)


@dataclass(frozen=True)
class Prediction:
    """What the model expects if Blue plays `plan`. `trajectory` has one row per 0.125 s, in game units."""

    plan: str
    trajectory: pd.DataFrame
    typical_error: dict | None  # held-out error at this horizon, per side; None past the tested horizons
    warnings: tuple[str, ...] = ()

    @property
    def horizon(self) -> float:
        return float(self.trajectory["t"].iloc[-1])

    @property
    def start(self) -> pd.Series:
        return self.trajectory.iloc[0]

    @property
    def end(self) -> pd.Series:
        return self.trajectory.iloc[-1]

    @property
    def margin(self) -> float:
        """Blue hp minus Red hp at the horizon: the score the planners use."""
        return float(self.end["hp_blue"] - self.end["hp_red"])

    @property
    def margin_gain(self) -> float:
        """Change in the margin from now to the horizon."""
        return self.margin - float(self.start["hp_blue"] - self.start["hp_red"])

    def to_text(self) -> str:
        a, b = self.start, self.end
        lines = [f"If Blue plays {self.plan}, in {self.horizon:g} s the model expects:"]
        for s in SIDES:
            lines.append(
                f"  {s.capitalize()} hp {a[f'hp_{s}']:.0f} -> {b[f'hp_{s}']:.0f}, "
                f"units {a[f'units_{s}']:.0f} -> {b[f'units_{s}']:.1f}"
            )
        lines.append(f"  distance {a['distance']:.1f} -> {b['distance']:.1f}")
        lines.append(f"  change in hp margin (Blue - Red): {self.margin_gain:+.0f}")
        if self.typical_error:
            e = self.typical_error
            lines.append(
                f"  typical error at this horizon: {e['hp']:.0f} hp and {e['units']:.1f} units per side, "
                f"{e['distance']:.1f} in distance"
            )
        lines += [f"  Warning: {w}" for w in self.warnings]
        return "\n".join(lines)


class WorldModel:
    """A frozen pipeline model with a small, unit-safe interface (see the module docstring)."""

    def __init__(self, name: str, pipeline: Pipeline, info: dict):
        self.name, self.pipeline, self.info = name, pipeline, info
        self.types = pipeline.L.types
        self.hp_per_unit = {t: float(v) for t, v in pipeline.L.hp.items()}

    def __repr__(self) -> str:
        return f"WorldModel({self.name!r}, types={self.types}, reliable to {self.reliable_horizon:g} s)"

    # ---- files ----
    @staticmethod
    def available() -> list[str]:
        return sorted(p.stem for p in MODEL_DIR.glob("*.json"))

    @classmethod
    def load(cls, name: str | Path) -> WorldModel:
        """A model in configs/world_models/ by name, or any file written by `save`."""
        path = Path(name) if str(name).endswith(".json") else MODEL_DIR / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"no world model {name!r}; available: {cls.available()}")
        d = json.loads(path.read_text())
        if d["format"] != FORMAT:
            raise ValueError(f"{path} has format {d['format']}, expected {FORMAT}")
        return cls(d["name"], _pipeline_from_dict(d["model"]), d["info"])

    def save(self, path: str | Path) -> None:
        d = {
            "format": FORMAT,
            "name": self.name,
            "info": self.info,
            "equations": self.equations().splitlines(),
            "model": _pipeline_to_dict(self.pipeline),
        }
        Path(path).write_text(json.dumps(d, indent=1) + "\n")

    # ---- states ----
    def state_from_units(self, units: pd.DataFrame | Iterable[dict]) -> State:
        """A state from one row per unit, from any simulator.

        Needs columns team ("blue" / "red"), unit_type (a SMAClite name, e.g. "MARINE"), x, y and hp;
        optional shield (0) and alive (hp > 0). Distances are in map units, centre to centre.
        """
        u = pd.DataFrame(units).copy()
        missing = {"team", "unit_type", "x", "y", "hp"} - set(u.columns)
        if missing:
            raise ValueError(f"units are missing the columns {sorted(missing)}")
        u["unit_type"] = u["unit_type"].map(lambda t: t.upper() if isinstance(t, str) else str(t))
        u["shield"] = u["shield"].fillna(0.0) if "shield" in u else 0.0
        if "alive" in u:
            if u["alive"].isna().any():
                raise ValueError("alive is missing for some units")
            u = u[u["alive"].astype(bool)].copy()
        try:
            u["hp"] = u["hp"].astype(float)
            if not np.isfinite(u["hp"]).all():
                raise ValueError
            u = u[u["hp"] > 0].copy()
            numbers = u[["x", "y", "hp", "shield"]].astype(float)
        except (TypeError, ValueError):
            raise ValueError("x, y, hp and shield must be finite numbers") from None
        u[["x", "y", "hp", "shield"]] = numbers
        if not set(u["team"]) <= set(SIDES):
            raise ValueError(f"team must be one of {SIDES}")
        unknown = set(u["unit_type"].str.lower()) - set(self.types)
        if unknown:
            raise ValueError(f"model {self.name!r} knows {self.types}, not {sorted(unknown)}")
        for s in SIDES:
            if not (u["team"] == s).any():
                raise ValueError(f"no {s} units alive: the battle is over")
        if not np.isfinite(numbers.to_numpy()).all():
            raise ValueError("x, y, hp and shield must be finite numbers")
        stats = {t: unit_stats(t.upper()) for t in self.types}
        for k in ("hp", "shield"):
            full = u["unit_type"].str.lower().map(lambda t, k=k: stats[t][k])
            if ((u[k] > full * 1.001) | (u[k] < 0)).any():
                raise ValueError(f"{k} must be between 0 and the unit type's maximum")
        u["unit_id"] = u.groupby("team").cumcount()
        # the measured fire columns are not used by the models; give `typed_states` no targets
        u = u.assign(step=0, alive=True, target_team=None, target_id=-1, target_in_range=False)
        types = tuple(t.upper() for t in self.types)
        row = typed_states(u, types, max_reach=self.pipeline.max_reach).iloc[0]
        return State(row, self.types, self.hp_per_unit)

    def observe(self, env) -> State:
        """A state from a live SMAClite environment."""
        game = env.unwrapped
        return self.state_from_units(
            [
                {
                    "team": team,
                    "unit_type": getattr(u.type, "name", str(u.type)),
                    "x": float(u.pos[0]),
                    "y": float(u.pos[1]),
                    "hp": float(u.hp),
                    "shield": float(u.shield),
                }
                for team, group in (("blue", game.agents), ("red", game.enemies))
                for u in group.values()
            ]
            or pd.DataFrame(columns=["team", "unit_type", "x", "y", "hp"], dtype=object)
        )

    # ---- predictions ----
    @property
    def reliable_horizon(self) -> float:
        """Longest horizon (s) that met the pre-registered rule on held-out battles (docs/world_model.md)."""
        return float(self.info["reliable_horizon_s"])

    def predict(self, state: State, plan: Plan, horizon: float | None = None) -> Prediction:
        """What happens if Blue plays `plan`: one intent ("hold/focus", for `horizon` seconds) or a sequence
        [("withdraw/nearest", 1.0), ("advance/focus", 2.0)]."""
        return self.predict_many(state, [plan], horizon)[0]

    def predict_many(
        self, state: State, plans: Sequence[Plan], horizon: float | None = None
    ) -> list[Prediction]:
        """`predict` for several plans from the same state. With a common horizon they run as one batch."""
        if not isinstance(state, State):
            raise ValueError("state must come from observe() or state_from_units()")
        if not len(plans):
            return []
        if state.types != self.types:
            raise ValueError(f"the state has types {state.types}, the model {self.types}")
        controls = [plan_controls(p, horizon) for p in plans]
        if len({len(U) for U in controls}) > 1:
            return [self._predict(state, U[None], [p])[0] for p, U in zip(plans, controls, strict=True)]
        return self._predict(state, np.stack(controls), plans)

    def _predict(self, state: State, U: np.ndarray, plans: Sequence[Plan]) -> list[Prediction]:
        L, n = self.pipeline.L, len(U)

        def rep(cols):
            v = np.nan_to_num(state.row[list(cols)].to_numpy(float))
            return np.repeat(v[None], n, 0)

        B = {
            "X0": rep(L.x_cols),
            "shape0": rep(SHAPE),
            "front0": rep(L.front_cols),
            "reach0": rep(L.r_cols),
            "U": U,
        }
        with np.errstate(invalid="ignore", divide="ignore"):
            X = self.pipeline.rollout(B)
        horizon = U.shape[1] * DT
        error, warnings = self.typical_error(horizon), []
        if horizon > self.reliable_horizon + 1e-9:
            warnings.append(
                f"horizon {horizon:g} s is past the {self.reliable_horizon:g} s the model is reliable to"
            )
        out = []
        for p, u, x in zip(plans, U, X, strict=True):
            tr = self._trajectory(x)
            w = [*warnings, STAND_OFF] if not u[:, 2].any() and _stand_off(tr) else warnings
            out.append(Prediction(plan_text(p), tr, error, tuple(w)))
        return out

    def _trajectory(self, x: np.ndarray) -> pd.DataFrame:
        L, out = self.pipeline.L, {"t": np.arange(len(x)) * DT}
        for s in SIDES:
            hp = {t: x[:, L.idx("health", s, t)] * REF_UNITS * self.hp_per_unit[t] for t in self.types}
            al = {t: x[:, L.idx("alive", s, t)] * REF_UNITS for t in self.types}
            out[f"hp_{s}"], out[f"units_{s}"] = sum(hp.values()), sum(al.values())
            if len(self.types) > 1:
                out |= {f"hp_{s}_{t}": v for t, v in hp.items()}
                out |= {f"units_{s}_{t}": v for t, v in al.items()}
        out["distance"] = x[:, -1]
        return pd.DataFrame(out)

    def typical_error(self, horizon: float) -> dict | None:
        """Mean absolute error per side on held-out battles, interpolated between the tested horizons
        (0 s counts as no error); None past the longest."""
        e = self.info["held_out_error"]
        hs = [0.0, *(float(h) for h in e["horizon_s"])]
        if horizon > hs[-1]:
            return None
        return {k: float(np.interp(horizon, hs, [0.0, *e[k]])) for k in ("hp", "units", "distance")}

    def compare(
        self,
        state: State,
        plans: Sequence[Plan] = INTENTS,
        horizon: float | None = None,
        commit: float = 2.0,
        then: Intent = "advance/nearest",
    ) -> pd.DataFrame:
        """Predicted outcome of each plan, best hp margin first.

        A single intent is played for `commit` seconds and then `then` to the horizon, the scheme of the
        planner experiments; a (intent, seconds) sequence is played as given. `horizon` defaults to the
        reliable horizon. Differences much smaller than `Prediction.typical_error` are near-ties, and the
        ranking was not found to beat the best fixed intent in general (docs/world_model.md).
        """
        horizon = self.reliable_horizon if horizon is None else horizon
        _steps(horizon), _steps(commit)
        full = []
        for p in plans:
            if _is_intent(p):
                rest = horizon - commit
                p = [(p, min(commit, horizon))] + ([(then, rest)] if rest >= DT else [])
            full.append(p)
        rows = [
            {
                "plan": plan_text(p),
                "margin": pr.margin,
                "margin_gain": pr.margin_gain,
                "hp_blue": pr.end["hp_blue"],
                "hp_red": pr.end["hp_red"],
                "units_blue": pr.end["units_blue"],
                "units_red": pr.end["units_red"],
                "distance": pr.end["distance"],
            }
            for p, pr in zip(plans, self.predict_many(state, full, horizon), strict=True)
        ]
        if not rows:
            raise ValueError("no plans to compare")
        df = pd.DataFrame(rows).sort_values("margin", ascending=False, kind="stable", ignore_index=True)
        df.insert(2, "behind_best", df["margin"].iloc[0] - df["margin"])
        return df

    def planner(self, **kwargs):
        """A Blue policy that plans with this model (`envs.planner.PipelinePlannerPolicy`)."""
        from sindy_wm.envs.planner import PipelinePlannerPolicy

        return PipelinePlannerPolicy(
            self.pipeline, **{"withdraw_radius": self.pipeline.withdraw_radius} | kwargs
        )

    def equations(self) -> str:
        """The model as readable lines, in its internal scales (docs/generalisation/06)."""
        m, side, lines = self.pipeline, {"blue": "B", "red": "R"}, []
        for (s, t), c in m.coef.items():
            terms = " ".join(f"{v:+.4f}·{k}" for k, v in c.items() if v)
            lines.append(f"H_{side[s]}^{t}' = {terms}")
        for (s, t), (p, pf) in m.p.items():
            f = f" {pf:+.3f}·focus" if pf else ""
            lines.append(f"(ln A_{side[s]}^{t})' = ({p:.3f}{f}) · (ln H_{side[s]}^{t})'")
        d = " ".join(f"{c:+.3f}·{n}" for n, c in zip(m.dist_names, m.dist_coef, strict=True) if c)
        lines.append(f"d' = {d}")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------- files
def _key(k: tuple[str, str]) -> str:
    return ":".join(k)


def _unkey(d: dict) -> dict:
    return {tuple(k.split(":")): v for k, v in d.items()}


def _pipeline_to_dict(m: Pipeline) -> dict:
    """Every fitted number of a pipeline, as JSON. Floats survive the round trip exactly."""
    return {
        "types": list(m.L.types),
        "spec": {k: getattr(m, k) for k in _SPEC},
        "rho": m.rho,
        "health": {_key(k): {n: float(v) for n, v in c.items()} for k, c in m.coef.items()},
        "alive": {_key(k): [float(v) for v in p] for k, p in m.p.items()},
        "distance": dict(zip(m.dist_names, [float(c) for c in m.dist_coef], strict=True)),
        "reach_closure": {_key(k): [float(v) for v in p] for k, p in m.share.items()},
        "target_choice": {_key(k): [float(v) for v in p] for k, p in m.alloc_b.items()},
        "team_closure": {
            "form": m.P_team["form"],
            "gates": {k: [float(v) for v in g] for k, g in m.P_team["gates"].items()},
            "share_blue": [float(v) for v in m.P_team["share_blue"]],
            "share_red": [float(v) for v in m.P_team["share_red"]],
        },
    }


def _pipeline_from_dict(d: dict) -> Pipeline:
    m = Pipeline(tuple(d["types"]), **d["spec"])
    m.rho = dict(d["rho"])
    m.coef = _unkey(d["health"])
    m.p = {k: tuple(v) for k, v in _unkey(d["alive"]).items()}
    m.dist_names, m.dist_coef = list(d["distance"]), np.array(list(d["distance"].values()))
    m.share = {k: np.array(v) for k, v in _unkey(d["reach_closure"]).items()}
    m.alloc_b = {k: np.array(v) for k, v in _unkey(d["target_choice"]).items()}
    t = d["team_closure"]
    m.P_team = {
        "form": t["form"],
        "gates": {k: np.array(v) for k, v in t["gates"].items()},
        "share_blue": np.array(t["share_blue"]),
        "share_red": np.array(t["share_red"]),
    }
    return m
