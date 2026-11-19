"""Step 1b: marauders alone (exp09). Does the marine model transfer, and does the recipe refit?

1. Transfer: A*-2k, A and A* fitted on exp06 `train` (marines), scaled to marauders without
   refitting (`switch_library.scaled`: reach, walking speed, time to kill), scored on exp09 `val`.
2. Refit: A*-2k's recipe fitted on exp09 `train`: closures with Red/Blue reach 7.125 and gate
   starting values scaled by the reach ratio, STLSQ (thr 0.15) on SG-9 rates, reach-regressor alive
   law. Compared with the game: firepower 19 / 1.125 s / 125 hp = 0.135 per unit in reach (tick-
   rounded cooldown), walking speed 3.15.
Both are scored with the harness's metrics. exp09 `test` is not read. Writes docs/mixed/02_marauders.md.

    uv run --extra explore experiments/mixed/marauders.py
"""

import copy
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sindy_wm.evaluation import harness as H
from sindy_wm.evaluation.explore_models import AliveVariantA, CombinedA
from sindy_wm.evaluation.harness_models import ModelA
from sindy_wm.models.intent_model import SHAPE, _close_flee_gate, _close_gate, fit_derived_share
from sindy_wm.models.switch_library import scaled
from sindy_wm.paths import PROJECT_ROOT

EXP = "exp09_marauders"
DOC = PROJECT_ROOT / "docs" / "mixed" / "02_marauders.md"
REACH_M, REACH_MRD = 5.75, 6 + 2 * 0.5625
GAME = {"k": 19 / (np.ceil(1.07 * 16) / 16) / 125, "speed": 3.15}


def md(table) -> str:
    return "```\n" + table.to_string() + "\n```"


def marauder_closures(rates: pd.DataFrame) -> dict:
    """A's closures on marauder data: gates started from the marine values scaled by the reach ratio."""
    s = REACH_MRD / REACH_M
    X = lambda a: tuple(a[c].to_numpy() for c in ["distance", *SHAPE])  # noqa: E731
    gates = {}
    for m, f, p0 in [
        ("advance", _close_gate, (6, 7.6 * s, 0, 0, 0, 0, 0.5 * s, 0.5)),
        ("hold", _close_gate, (3, 8 * s, 0, 0, 0, 0, 0.5 * s, 0.5)),
        ("withdraw", _close_flee_gate, (3, 8.5 * s, 0, 0, 0, 0, 0.3 * s, 0.5, 2, 6 * s, 0.5 * s, 0)),
    ]:
        a = rates[rates["movement"] == m]
        a = a.sample(min(100_000, len(a)), random_state=0)
        gates[m], _ = curve_fit(f, X(a), a["d_distance"], p0=p0, maxfev=80000)
    return {
        "gates": gates,
        "share_blue": fit_derived_share(rates, "blue", rho=REACH_MRD)[0],
        "share_red": fit_derived_share(rates, "red", rho=REACH_MRD)[0],
        "form": "derived",
    }


def score(Y_h, Bh, Y_b, Bb) -> dict:
    return {**H.horizon_metrics(Y_h, Bh), **H.battle_metrics(Y_b, Bb)}


def roll(model, B, transfer: bool):
    m = scaled(model.m, "marine", "marauder") if transfer else model.m
    return m.simulate(B["X0"], B["shape0"], B["U"], reach0=B["reach0"])


def main() -> None:
    val = H.load_states("val", EXP)
    Bh, Bb = H.horizon_batch(val), H.battle_batch(val)
    marine = H.FitData(states=None, rates=H.both_alive(H.rows_with_rates("train")))
    raw = H.rows_with_rates("train", experiment=EXP)
    mrd = H.FitData(states=None, rates=H.both_alive(raw), raw_rates=raw)
    rows, coefs = {}, {}
    persistence = np.repeat(Bh["X0"][:, None].astype(float), Bh["U"].shape[1] + 1, 1)
    persistence_b = np.repeat(Bb["X0"][:, None].astype(float), Bb["U"].shape[1] + 1, 1)
    rows["persistence"] = score(persistence, Bh, persistence_b, Bb)
    for name, cls in (("A", ModelA), ("A*", CombinedA), ("A*-2k", lambda: AliveVariantA(law="reach"))):
        m = cls().fit(marine)
        rows[f"{name}, marines scaled"] = score(roll(m, Bh, True), Bh, roll(m, Bb, True), Bb)
        coefs[f"{name}, marines scaled"] = scaled(m.m, "marine", "marauder")
    P = marauder_closures(mrd.rates)
    refit = AliveVariantA(law="reach", P=copy.deepcopy(P)).fit(mrd)
    rows["A*-2k, refit on marauders"] = score(roll(refit, Bh, False), Bh, roll(refit, Bb, False), Bb)
    coefs["A*-2k, refit on marauders"] = refit.m
    pickle.dump(refit, open(H.CACHE / "marauders_Astar2k.pkl", "wb"))

    table = pd.DataFrame(rows).T
    cols = [
        "health_1s",
        "health_2s",
        "health_4s",
        "margin_r2_2s",
        "margin_r2_4s",
        "alive_4s",
        "distance_4s",
        "battle_health_nrmse",
        "winner_acc",
        "duration_err_s",
        "blowup_rate",
    ]
    terms = {}
    for name, m in coefs.items():
        t = {}
        for eq in ("health_blue", "health_red", "distance"):
            for n, c in zip(m.names, m.coef[eq], strict=True):
                if c:
                    t[f"{eq}: {n}"] = c
        for side in ("blue", "red"):
            for k, v in m.p[side].items():
                if v:
                    t[f"alive_{side}: {k}"] = v
        terms[name] = t
    terms = pd.DataFrame(terms)
    lines = [
        "# Step 1b: marauders alone (exp09, 19 Nov)",
        "",
        "Script: `experiments/mixed/marauders.py`. Scored on exp09 `val` (500 battles); `test` not read.",
        "Health scale: 1 = ten full-health marauders (125 hp each). Game values: firepower",
        f"{GAME['k']:.3f} per marauder in reach (19 dmg per 1.125 s on 125 hp), walking speed 3.15.",
        "",
        "## Prediction (harness metrics)",
        "",
        md(table[cols].round(3)),
        "",
        "## Coefficients (scaled marine models, and the refit)",
        "",
        md(terms.round(4)),
        "",
    ]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
