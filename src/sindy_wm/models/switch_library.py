"""Switch candidates chosen by sparse selection, and unit-type scaling of the intent model.

Two ideas on top of `intent_model`:

1. **Switches as library candidates.** Instead of fitting the reach closure and the movement
   gates with `curve_fit`, each mechanism gets a family of candidate switches (normal CDFs at
   distances c on a grid, and widths w for the movement gates) and a best-subset selection keeps
   one per mechanism. The geometry inside the switch has no fitted parameters:

   - fire: share of a side's units within c of the enemy front, Phi((c - e) / depth_own), with
     e = sqrt(d^2 + (w_own - w_opp)^2) - depth_opp
   - movement: the gap between the fronts, d - depth_blue - depth_red, passes c

   The chosen switches are written in the `IntentModel` closure format, so the rest of the
   model (STLSQ per equation, simulation) is unchanged.

2. **Dimensionless form.** Distances in units of reach, time in units of how long one unit
   takes to kill another, speeds in units of walking speed. `scaled` rewrites a fitted model
   for another unit type without refitting.
"""

import copy

import numpy as np
import pandas as pd
from scipy.special import ndtr

from sindy_wm.models.intent_model import r2

C_GRID = np.round(np.arange(3.0, 14.01, 0.2), 2)
W_GRID = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
MOVES = ("advance", "hold", "withdraw", "flee")

# reach (attack range + two unit radii), walking speed, health per unit (hp + shield), and health
# lost per second to one shooting unit of the same type (armor and bonuses included)
UNITS = {
    "marine": {"reach": 5 + 2 * 0.375, "speed": 3.15, "health": 45.0, "dps": 6 / 0.61},
    "stalker": {"reach": 6 + 2 * 0.625, "speed": 4.13, "health": 160.0, "dps": 17.5 / 1.34},
}
for _u in UNITS.values():
    _u["tau"] = _u["health"] / _u["dps"]  # seconds for one unit to kill one unit


def fire_switch(d: pd.DataFrame, side: str, c: float) -> np.ndarray:
    """Units of `side` within c of the enemy front, and its products with withdraw, hold, focus."""
    opp = "red" if side == "blue" else "blue"
    e = np.sqrt(d["distance"] ** 2 + (d[f"width_{side}"] - d[f"width_{opp}"]) ** 2) - d[f"depth_{opp}"]
    s = (d[f"alive_{side}"] * ndtr((c - e) / np.sqrt(0.05**2 + d[f"depth_{side}"] ** 2 + 1e-4))).to_numpy()
    return np.column_stack(
        [s, s * d["withdraw"].to_numpy(), s * d["hold"].to_numpy(), s * d["focus"].to_numpy()]
    )


def move_switches(d: pd.DataFrame, cs: dict, ws: dict) -> np.ndarray:
    """Closing switches per movement (front gap above c) and the withdraw flee switch."""
    gap = (d["distance"] - d["depth_blue"] - d["depth_red"]).to_numpy()
    F = [d[m].to_numpy() * ndtr((gap - cs[m]) / ws[m]) for m in MOVES[:3]]
    flee = (cs["flee"] + d["depth_blue"].to_numpy() - d["distance"].to_numpy()) / ws["flee"]
    F.append(d["withdraw"].to_numpy() * ndtr(flee))
    return np.column_stack(F)


def _fit_r2(T, y):
    k = np.linalg.lstsq(T, y, rcond=None)[0]
    return r2(y, T @ k)


def fire_profile(rows: pd.DataFrame, side: str, grid=C_GRID) -> pd.Series:
    """R² of the losses `side` inflicts, with one switch at each candidate c."""
    target = rows["d_health_red" if side == "blue" else "d_health_blue"].to_numpy()
    return pd.Series([_fit_r2(fire_switch(rows, side, c), target) for c in grid], index=grid, name=side)


def choose_switches(
    rows: pd.DataFrame, n: int = 200_000, sweeps: int = 3, seed: int = 0
) -> tuple[dict, dict]:
    """Best-subset choice of one switch per mechanism (training rows only).

    Returns the closures in `IntentModel` format and the chosen values.
    """
    sub = rows.sample(min(n, len(rows)), random_state=seed)
    fire = {side: fire_profile(sub, side).idxmax() for side in ("blue", "red")}
    cs, ws = dict.fromkeys(MOVES, 7.0), dict.fromkeys(MOVES, 0.5)
    y = sub["d_distance"].to_numpy()
    for _ in range(sweeps):  # coordinate-wise: one mechanism at a time, the others fixed
        for m in MOVES:
            _, cs[m], ws[m] = max(
                (_fit_r2(move_switches(sub, cs | {m: c}, ws | {m: w}), y), c, w)
                for c in C_GRID
                for w in W_GRID
            )
    P = {
        "share_blue": np.array([fire["blue"], 0, 1, 0.05, 1, 1.0]),
        "share_red": np.array([fire["red"], 0, 1, 0.05, 1, 1.0]),
        "gates": {m: np.array([0, cs[m], 1, 1, 0, 0, ws[m], 0]) for m in ("advance", "hold")},
    }
    P["gates"]["withdraw"] = np.array(
        [0, cs["withdraw"], 1, 1, 0, 0, ws["withdraw"], 0, 0, cs["flee"], ws["flee"], 1]
    )
    chosen = (
        {f"fire {s}": c for s, c in fire.items()}
        | {f"{m} c": cs[m] for m in MOVES}
        | {f"{m} w": ws[m] for m in MOVES}
    )
    return P, chosen


def dimensionless(model, unit: str) -> pd.Series:
    """Coefficients in unit-free form: firepower x tau, speeds / walking speed, switches / reach."""
    u = UNITS[unit]
    out = {}
    for eq in ("health_blue", "health_red", "distance"):
        f = u["tau"] if eq != "distance" else 1 / u["speed"]
        out |= {f"{eq}: {n}": c * f for n, c in zip(model.names, model.coef[eq], strict=True) if c != 0}
    out |= {f"fire switch {s} / reach": model.P[f"share_{s}"][0] / u["reach"] for s in ("blue", "red")}
    out |= {f"{g} switch / reach": model.P["gates"][g][1] / u["reach"] for g in model.P["gates"]}
    out |= {f"alive p_{s}: {k}": v for s in ("blue", "red") for k, v in model.p[s].items() if v != 0}
    return pd.Series(out)


def scaled(model, source: str, target: str):
    """A copy of `model` rewritten for another unit type, without refitting.

    Switch distances and widths scale with reach, firepower coefficients with 1 / tau, closing
    speeds with walking speed. Dimensionless parameters (depth shifts, alive exponents) stay.
    """
    a, b = UNITS[source], UNITS[target]
    s, ft, fv = b["reach"] / a["reach"], a["tau"] / b["tau"], b["speed"] / a["speed"]
    m = copy.deepcopy(model)
    for side in ("share_blue", "share_red"):
        m.P[side] = np.array(m.P[side], float)
        m.P[side][[0, 3]] *= s
    m.P["gates"] = {g: np.array(v, float) for g, v in m.P["gates"].items()}
    for v in m.P["gates"].values():
        v[[1, 6]] *= s
    m.P["gates"]["withdraw"][[9, 10]] *= s
    for eq in ("health_blue", "health_red"):
        m.coef[eq] = m.coef[eq] * ft
    m.coef["distance"] = m.coef["distance"] * fv
    return m
