"""Withdraw in the distance equation: fitted gates or the scripts' rule? (val only, 21 Nov)

After the WP6 dry run (refit on 500 marauder val battles gave a withdraw closing speed of 12.8), compare the
pipeline's two ways of handling withdraw *movement* (firing under withdraw: the frozen rule in both):

  gates  A's fitted withdraw switches: close*wd, flee*wd (a 12-number curve_fit closure)
  rule   the scripts' rule, no switch: danger*wd (Blue within the withdraw radius walks away) and walk_r*wd
         (Red not in reach walks in); expected coefficients +3.15 and -3.15 (walking speed)

For each scenario: val metrics (distance at 1 / 2 / 4 s overall and on start points withdrawing > 50% of the
first 2 s; health, margin, battle nRMSE) and the withdraw coefficients from refits on small sets (val, and two
random 500-battle subsets of train). Writes docs/generalisation/05_withdraw_movement.md.

    uv run --extra explore experiments/generalisation/withdraw_movement.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DOCS, SCENARIOS, layout, md, rows, states  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.models.intent_model import DT, add_controls  # noqa: E402
from sindy_wm.models.pipeline import Pipeline  # noqa: E402

RADIUS = {"marines": 7.0, "marauders": 7.0 * (6 + 2 * 0.5625) / 5.75, "mixed": 8.0}  # the scripts' settings
WD_TERMS = ("close*wd", "flee*wd", "danger*wd", "walk_r*wd")


def wd_coefs(m: Pipeline) -> dict:
    return {n: float(c) for n, c in zip(m.dist_names, m.dist_coef, strict=True) if c and n in WD_TERMS}


def dist_terms(m: Pipeline) -> dict:
    return {n: round(float(c), 3) for n, c in zip(m.dist_names, m.dist_coef, strict=True) if c}


def subset(r: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    eps = r.index.get_level_values(0).unique().to_numpy()
    pick = np.random.default_rng(seed).choice(eps, min(n, len(eps)), replace=False)
    return r[r.index.get_level_values(0).isin(pick)]


def main() -> None:
    lines = [
        "# Withdraw movement: fitted gates or the scripts' rule (val, 21 Nov)",
        "",
        "Script: `experiments/generalisation/withdraw_movement.py`. Firing under withdraw: the frozen rule;",
        "target choice: proportional. Distance in map units; health in ten units of the first type.",
        "",
    ]
    for scen in SCENARIOS:
        L = layout(scen)
        types = SCENARIOS[scen][1]
        tr, vr = rows(scen, "train"), rows(scen, "val")
        val = add_controls(states(scen, "val"))
        Bh, Bb = L.batch(val), L.battle_batch(val)
        wd_share = Bh["U"][:, : round(2 / DT), 2].mean(1)
        res, coefs, terms = {}, {}, {}
        for mv in ("gates", "rule"):
            kw = {"withdraw": "rule", "alloc": "rule", "movement_wd": mv, "withdraw_radius": RADIUS[scen]}
            m = Pipeline(types, **kw).fit(tr)
            Yh, Yb = L.team_view(m.rollout(Bh)), L.team_view(m.rollout(Bb))
            met = {**H.horizon_metrics(Yh, Bh["team"]), **H.battle_metrics(Yb, Bb["team"])}
            for h in (1, 2, 4):
                n = round(h / DT)
                ok = ~np.isnan(Bh["team"]["Y"][:, n, 4]) & (wd_share > 0.5)
                met[f"distance_{h}s, withdrawing"] = float(
                    np.abs(Yh[ok, n, 4] - Bh["team"]["Y"][ok, n, 4]).mean()
                )
            res[mv] = met
            terms[mv] = dist_terms(m)
            coefs[(mv, "train (full)")] = wd_coefs(m)
            for label, r in (
                ("val (refit)", vr),
                ("train, 500 battles (a)", subset(tr, 500, 1)),
                ("train, 500 battles (b)", subset(tr, 500, 2)),
            ):
                try:
                    coefs[(mv, label)] = wd_coefs(Pipeline(types, **kw).fit(r))
                except RuntimeError as err:  # a closure fit that does not converge is itself a result
                    coefs[(mv, label)] = {"fit failed": str(err)[:60]}
        cols = [
            "distance_1s",
            "distance_2s",
            "distance_4s",
            "distance_1s, withdrawing",
            "distance_2s, withdrawing",
            "distance_4s, withdrawing",
            "health_2s",
            "health_4s",
            "margin_r2_2s",
            "battle_health_nrmse",
            "duration_err_s",
        ]
        lines += [
            f"## {scen} (withdraw radius {RADIUS[scen]:.2f})",
            "",
            f"Start points withdrawing > 50% of the first 2 s: {int((wd_share > 0.5).sum())} "
            f"of {len(wd_share)}.",
            "",
            md(pd.DataFrame(res).T[cols].round(4)),
            "",
            "Distance equation (train, full):",
            "",
        ]
        lines += [f"- {mv}: {t}" for mv, t in terms.items()]
        C = pd.DataFrame(coefs).T.round(3)
        lines += ["", "Withdraw coefficients across refits:", "", md(C), ""]
        print(scen, "done", flush=True)
    (DOCS / "05_withdraw_movement.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
