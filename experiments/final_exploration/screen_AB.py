"""Axes A and B, screening stage: do per-unit features carry information A's state lacks?

An oracle, one-step screen, so an upper bound. Each candidate term uses the *measured* feature and
is added to A's selected terms for one equation. The score is the held-out R² of the SG-9 rate,
over the same 5 folds of `train` as the harness (least squares on the fold's training battles, R² on
its held-out battles). A candidate that gains nothing here cannot help a rollout model either. One
that gains still needs an equation of its own before it can be used (the next stage).

Equations and their base terms (A's selection):
    health_blue   R_R                       health_red   R_B, R_B*wd
    alive_<side>  A (ln H)'  (A's death law, rewritten as A' = p A (ln H)')

Candidates (own = the equation's side, enemy = the other side):
    health:  enemy firing F, volley V (and V smoothed over a volley period), enemy overkill, own engaged
             health, own front-half health, own / enemy dispersion (rg), hull, largest cluster,
             nearest-enemy distance, velocity towards the enemy; F replacing R ("firing as the state")
    alive:   own units about to die x enemy firing, own Gini x (ln H)', own min health x (ln H)'

    uv run --extra explore experiments/final_exploration/screen_AB.py
Writes runs/final_exploration/screen_AB.csv (one row per equation x candidate x fold).
"""

import numpy as np
import pandas as pd

from sindy_wm.evaluation import harness as H

OUT = H.OUT / "screen_AB.csv"


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def load() -> pd.DataFrame:
    rows = H.both_alive(H.rows_with_rates("train"))
    f = pd.read_parquet(H.CACHE / "features_train.parquet")
    d = rows.join(f, how="inner")
    # the volley rhythm is 5 steps: a centred 5-step mean, per battle
    for side in ("blue", "red"):
        d[f"volley5_{side}"] = d.groupby(level=0)[f"volley_{side}"].transform(
            lambda v: v.rolling(5, center=True, min_periods=1).mean()
        )
    return d


def designs(d: pd.DataFrame) -> dict:
    """{equation: (y, base columns dict, {candidate: extra columns dict or ("replace", cols)})}."""
    out = {}
    for side, enemy in (("blue", "red"), ("red", "blue")):
        R_enemy = d[f"reach_{enemy}"]
        base = {"R_enemy": R_enemy}
        if side == "red":
            base["R_enemy*wd"] = R_enemy * d["withdraw"]
        F, V, V5 = d[f"firing_{enemy}"], d[f"volley_{enemy}"], d[f"volley5_{enemy}"]
        cands = {
            "F_enemy": {"F_enemy": F},
            "V_enemy": {"V_enemy": V},
            "V5_enemy": {"V5_enemy": V5},
            "overkill_enemy": {"overkill_enemy": d[f"overkill_{enemy}"]},
            "heng_own": {"heng_own": d[f"heng_{side}"]},
            "hfront_own": {"hfront_own": d[f"hfront_{side}"]},
            "rg_own*R": {"rg_own*R": d[f"rg_{side}"] * R_enemy},
            "rg_enemy*R": {"rg_enemy*R": d[f"rg_{enemy}"] * R_enemy},
            "hull_enemy*R": {"hull_enemy*R": d[f"hull_{enemy}"] * R_enemy},
            "cluster_enemy*R": {"cluster_enemy*R": d[f"cluster_{enemy}"] * R_enemy},
            "nnmean_own*R": {"nnmean_own*R": d[f"nnmean_{side}"] * R_enemy},
            "vto_enemy*R": {"vto_enemy*R": d[f"vto_{enemy}"] * R_enemy},
            "gini_own*R": {"gini_own*R": d[f"gini_{side}"] * R_enemy},
            "F replaces R": (
                "replace",
                {"F_enemy": F} | ({"F_enemy*wd": F * d["withdraw"]} if side == "red" else {}),
            ),
        }
        out[f"health_{side}"] = (d[f"d_health_{side}"], base, cands)
        A, H_, dH = d[f"alive_{side}"], d[f"health_{side}"], d[f"d_health_{side}"]
        dlnH = dH / H_.clip(lower=0.03)
        abase = {"A dlnH": A * dlnH}
        if side == "red":
            abase["A dlnH focus"] = A * dlnH * d["focus"]
        acands = {
            "dying_own*F_enemy": {"dying*F": d[f"dying_{side}"] * F},
            "dying_own*V5_enemy": {"dying*V5": d[f"dying_{side}"] * V5},
            "gini_own*A dlnH": {"gini*A dlnH": d[f"gini_{side}"] * A * dlnH},
            "hmin_own*A dlnH": {"hmin*A dlnH": d[f"hmin_{side}"] * A * dlnH},
            "overkill_enemy": {"overkill_enemy": d[f"overkill_{enemy}"]},
        }
        out[f"alive_{side}"] = (d[f"d_alive_{side}"], abase, acands)
    return out


def main() -> None:
    d = load()
    eps = H.episodes(d)
    folds = H.folds(H.episodes(H.load_states("train")))
    rows = []
    for eq, (y, base, cands) in designs(d).items():
        variants = {"base": base}
        for name, extra in cands.items():
            variants[name] = extra[1] if isinstance(extra, tuple) else base | extra
        for k, held in enumerate(folds):
            test = np.isin(d.index.get_level_values(0), held)
            for name, cols in variants.items():
                X = np.column_stack([np.asarray(v, float) for v in cols.values()])
                c, *_ = np.linalg.lstsq(X[~test], y.to_numpy()[~test], rcond=None)
                rows.append(
                    {
                        "equation": eq,
                        "candidate": name,
                        "fold": k,
                        "r2_heldout": r2(y.to_numpy()[test], X[test] @ c),
                        "coef": "; ".join(f"{n} {v:+.4f}" for n, v in zip(cols, c, strict=True)),
                    }
                )
    res = pd.DataFrame(rows)
    H.OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    base = res[res["candidate"] == "base"].set_index(["equation", "fold"])["r2_heldout"]
    res["gain"] = (
        res["r2_heldout"] - base.reindex(pd.MultiIndex.from_frame(res[["equation", "fold"]])).to_numpy()
    )
    s = res.groupby(["equation", "candidate"])["gain"].agg(["mean", "std"]).round(4)
    s["r2"] = res.groupby(["equation", "candidate"])["r2_heldout"].mean().round(3)
    pd.set_option("display.width", 200)
    for _eq, g in s.groupby(level=0):
        print(g.sort_values("mean", ascending=False).to_string(), "\n")
    print("battles:", len(eps))


if __name__ == "__main__":
    main()
