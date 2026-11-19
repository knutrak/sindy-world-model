"""Checkpoint 2: does SINDy recover the 2x2 damage matrix one step ahead, with measured reach and
measured fire allocation? (exp08 train; scored on val. No closures, no rollouts.)

For each side s and target type t, the health rate of t (SG 9, per type in units of ten full units) is
fitted by STLSQ (threshold 0.15, standardised features) on three libraries:

  (i)   measured fire:   F_{a->t} (enemy units of type a shooting a t in range), plus distractors:
                          fire on the other type, enemy reach and alive per type, a constant
  (ii)  reach x share:   R_a * sigma_{a,t} (sigma = measured share of a's fire on t), + R_a*sigma*wd
                          for Blue attackers, plus distractors (R_a * sigma_{a,t'}, alive, constant).
                          This is the form a rollout model uses: R from a reach closure, sigma from an
                          allocation closure. k = rate per firing unit x share of reach firing.
  (iii) reach x rule:    as (ii), but sigma_{a,t} = the share of type t among the target side's alive
                          units (the proportional allocation rule, no measurement)

and compared with (iv) a team-level fit on total hp (one k per side, A*-2k's health equation) via
the one-step R^2 of the total health rate. Per-type alive law (A*-2k's reach-regressor form):
A_t' = p_t * Hhat_t' * A_t / H_t, Hhat from (ii) with measured inputs, Red with a focus term.
Battle bootstrap (20 resamples) for the (ii) coefficients. Writes docs/mixed/03_oracle.md.

    uv run --extra explore experiments/mixed/oracle.py
"""

import numpy as np
import pandas as pd

from sindy_wm.models.intent_model import add_controls, add_rates, r2, stlsq
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT
from sindy_wm.states.typed import unit_stats

P = DATA_DIR / "processed" / "exp08_mixed"
DOC = PROJECT_ROOT / "docs" / "mixed" / "03_oracle.md"
TYPES = ("marine", "marauder")
SIDES = (("blue", "red"), ("red", "blue"))
THR = 0.15
FIRING = {"marine": 0.79, "marauder": 0.54}  # share of units in reach firing, single-type data (step 1b)


def configure(max_reach: float | None) -> str:
    """Use the reach-capped tables (exp08_mixed_r6) if max_reach is given; returns the file suffix."""
    global P, DOC
    if max_reach is None:
        return ""
    suffix = f"_r{max_reach:g}"
    P = DATA_DIR / "processed" / f"exp08_mixed{suffix}"
    DOC = DOC.with_name(DOC.stem + suffix + ".md")
    return suffix


def md(table) -> str:
    return "```\n" + table.to_string() + "\n```"


def game_rate(a: str, t: str) -> float:
    sa, st = unit_stats(a.upper()), unit_stats(t.upper())
    bonus = 10.0 if (a == "marauder" and t == "marauder") else 0.0
    return (
        (sa["damage"] + bonus - st["armor"]) / (np.ceil(sa["cooldown"] * 16) / 16) / (st["hp"] + st["shield"])
    )


def rates(name: str) -> pd.DataFrame:
    path = P / f"rates_{name}_w9.parquet"
    if path.exists():
        return pd.read_parquet(path)
    s = add_controls(pd.read_parquet(P / f"{name}.parquet"))
    hp = {t: unit_stats(t.upper())["hp"] for t in TYPES}
    for side, _ in SIDES:
        s[f"hp_{side}"] = sum(s[f"health_{side}_{t}"] * 10 * hp[t] for t in TYPES)
        s[f"reach_{side}"] = sum(s[f"reach_{side}_{t}"] for t in TYPES)
        s[f"alive_{side}"] = sum(s[f"alive_{side}_{t}"] for t in TYPES)
    cols = [f"{q}_{side}_{t}" for q in ("health", "alive") for side, _ in SIDES for t in TYPES]
    cols += [f"hp_{side}" for side, _ in SIDES] + [f"alive_{side}" for side, _ in SIDES] + ["distance"]
    r = add_rates(s, cols=tuple(cols))
    r = r[(r["alive_blue"] > 0) & (r["alive_red"] > 0)]
    r.to_parquet(path)
    return r


def sigma(r: pd.DataFrame, side: str, a: str, t: str, rule: bool) -> np.ndarray:
    """Share of attacker type a's (on `side`) fire that lands on target type t."""
    other = "red" if side == "blue" else "blue"
    alive = sum(r[f"alive_{other}_{x}"] for x in TYPES)
    prop = np.divide(r[f"alive_{other}_{t}"], alive, out=np.zeros(len(r)), where=alive > 0)
    if rule:
        return np.asarray(prop)
    f = r[f"fire_{side}_{a}_{t}"].to_numpy()
    tot = sum(r[f"fire_{side}_{a}_{x}"].to_numpy() for x in TYPES)
    return np.where(tot > 0, np.divide(f, tot, out=np.zeros(len(r)), where=tot > 0), prop)


def library(r: pd.DataFrame, side: str, t: str, kind: str) -> dict:
    """Candidate terms for the health rate of `side`'s type t (attackers are the other side)."""
    o = "red" if side == "blue" else "blue"
    t2 = [x for x in TYPES if x != t][0]
    wd = r["withdraw"].to_numpy() if o == "blue" else 0 * r["withdraw"].to_numpy()
    L = {}
    for a in TYPES:
        if kind == "fire":
            L[f"F[{a}->{t}]"] = r[f"fire_{o}_{a}_{t}"].to_numpy()
            L[f"F[{a}->{t2}]"] = r[f"fire_{o}_{a}_{t2}"].to_numpy()
            L[f"R[{a}]"] = r[f"reach_{o}_{a}"].to_numpy()
        else:
            s_t, s_t2 = sigma(r, o, a, t, kind == "rule"), sigma(r, o, a, t2, kind == "rule")
            R = r[f"reach_{o}_{a}"].to_numpy()
            L[f"R[{a}]*s[{t}]"] = R * s_t
            L[f"R[{a}]*s[{t2}]"] = R * s_t2
            if o == "blue":
                L[f"R[{a}]*s[{t}]*wd"] = R * s_t * wd
        L[f"A[{a}]"] = r[f"alive_{o}_{a}"].to_numpy()
    L["1"] = np.ones(len(r))
    return L


def fit(r, side, t, kind):
    L = library(r, side, t, kind)
    X = np.column_stack(list(L.values()))
    c = stlsq(X, r[f"d_health_{side}_{t}"].to_numpy(), THR)
    return dict(zip(L, c, strict=True))


def predict(r, side, t, kind, coef):
    L = library(r, side, t, kind)
    return sum(coef[k] * v for k, v in L.items())


def resample(r: pd.DataFrame, seed: int) -> pd.DataFrame:
    eps = r.index.get_level_values(0).unique().to_numpy()
    pick = np.random.default_rng(seed).choice(eps, len(eps), replace=True)
    return pd.concat([r.loc[[e]] for e in pick])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reach", type=float, default=None)
    configure(parser.parse_args().max_reach)
    tr, va = rates("train"), rates("val")
    lines = ["# Checkpoint 2: one-step damage matrix with measured reach and allocation (19 Nov)", ""]
    lines += [
        "Script: `experiments/mixed/oracle.py`. Fitted on exp08 `train` (6000 battles); one-step R² on",
        "`val`.",
        "STLSQ threshold 0.15 on standardised features. Units: target units removed per second (health",
        "per type in units of ten full units).",
        "",
    ]
    fits, rows = {}, []
    for kind in ("fire", "share", "rule"):
        for side, _o in SIDES:
            for t in TYPES:
                c = fits[(kind, side, t)] = fit(tr, side, t, kind)
                rows.append(
                    {
                        "library": kind,
                        "target": f"{t} ({side})",
                        "kept terms": ", ".join(f"{k} {v:+.4f}" for k, v in c.items() if v),
                        "val R²": r2(va[f"d_health_{side}_{t}"], predict(va, side, t, kind, c)),
                    }
                )
    lines += [
        "## Kept terms and one-step R² (val)",
        "",
        md(pd.DataFrame(rows).set_index(["library", "target"])),
        "",
    ]

    # matrix vs game
    mat = []
    for side, o in SIDES:
        for a in TYPES:
            for t in TYPES:
                g = game_rate(a, t)
                ci = fits[("fire", side, t)].get(f"F[{a}->{t}]", 0.0)
                cs = fits[("share", side, t)].get(f"R[{a}]*s[{t}]", 0.0)
                cr = fits[("rule", side, t)].get(f"R[{a}]*s[{t}]", 0.0)
                mat.append(
                    {
                        "attacker": f"{a} ({o})",
                        "target": t,
                        "game": g,
                        "(i) per firing unit": -ci,
                        "(i) / game": -ci / g,
                        "(ii) per unit in reach": -cs,
                        "(ii) / game": -cs / g,
                        "(ii) / (game x firing share 1b)": -cs / (g * FIRING[a]),
                        "(iii) per unit in reach": -cr,
                    }
                )
    lines += [
        "## The damage matrix against the game",
        "",
        "(i) measured fire; (ii) reach × measured allocation share; (iii) reach × proportional rule.",
        "The firing share from step 1b (single-type data): marines 0.79, marauders 0.54.",
        "",
        md(pd.DataFrame(mat).set_index(["attacker", "target"]).round(3)),
        "",
    ]

    # team-level comparison on total hp
    comp = []
    for side, o in SIDES:
        y_tr, y_va = tr[f"d_hp_{side}"].to_numpy(), va[f"d_hp_{side}"].to_numpy()
        cols = [f"reach_{o}"] + ([] if o == "red" else ["wd_reach"])
        for r_ in (tr, va):
            r_["wd_reach"] = r_["reach_blue"] * r_["withdraw"]
        k, *_ = np.linalg.lstsq(tr[cols].to_numpy(), y_tr, rcond=None)
        team = va[cols].to_numpy() @ k
        hp = {t: unit_stats(t.upper())["hp"] for t in TYPES}
        row = {"side": side, "team-level (one k, total hp)": r2(y_va, team)}
        for kind in ("fire", "share", "rule"):
            pred = sum(predict(va, side, t, kind, fits[(kind, side, t)]) * 10 * hp[t] for t in TYPES)
            row[f"per type ({kind})"] = r2(y_va, pred)
        comp.append(row)
    lines += [
        "## Does the per-type state help? One-step R² of the *total* health rate (hp/s), val",
        "",
        "The team-level fit is A*-2k's health equation on total hp with total reach (measured).",
        "",
        md(pd.DataFrame(comp).set_index("side").round(3)),
        "",
    ]

    # alive law per type
    al = []
    for side, _o in SIDES:
        for t in TYPES:
            for name, r_ in (("train", tr), ("val", va)):
                h = r_[f"health_{side}_{t}"].to_numpy()
                ok = (h > 0.01) & (r_[f"alive_{side}_{t}"].to_numpy() > 0)
                x = (
                    predict(r_, side, t, "share", fits[("share", side, t)])
                    * r_[f"alive_{side}_{t}"]
                    / np.maximum(h, 1e-9)
                ).to_numpy()
                X = np.column_stack([x, x * r_["focus"].to_numpy()]) if side == "red" else x[:, None]
                y = r_[f"d_alive_{side}_{t}"].to_numpy()
                if name == "train":
                    c, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
                else:
                    al.append(
                        {
                            "units": f"{t} ({side})",
                            "p": c[0],
                            "p_focus": c[1] if side == "red" else np.nan,
                            "val R² (alive rate)": r2(y[ok], X[ok] @ c),
                        }
                    )
    lines += [
        "## Alive law per type (A*-2k's form, Hhat from (ii))",
        "",
        "Single-type reference: marines p_B 0.66, p_R 0.54 (focus +0.24); marauders 0.63, 0.53 (+0.15).",
        "",
        md(pd.DataFrame(al).set_index("units").round(3)),
        "",
    ]

    # bootstrap of (ii)
    boots = []
    for b in range(20):
        rb = resample(tr, 500 + b)
        rec = {}
        for side, _o in SIDES:
            for t in TYPES:
                c = fit(rb, side, t, "share")
                for k, v in c.items():
                    rec[(f"{t} ({side})", k)] = v
        boots.append(rec)
    B = pd.DataFrame(boots)
    full = {
        (f"{t} ({side})", k): v
        for side, _ in SIDES
        for t in TYPES
        for k, v in fits[("share", side, t)].items()
    }
    summ = pd.DataFrame(
        {
            "value": pd.Series(full),
            "kept": (B != 0).mean(),
            "q2.5": B.quantile(0.025),
            "q97.5": B.quantile(0.975),
        }
    )
    summ = summ[(summ["value"] != 0) | (summ["kept"] > 0)]
    lines += ["## (ii) coefficients: 20 battle-bootstrap refits", "", md(summ.round(4)), ""]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
