# ruff: noqa: E501, E731, B007, B023, B905, I001
"""Exploratory, val only (docs/triple/00_plan.md): the triple model unchanged on the other scenarios, against each
scenario's own frozen model; and stalkers alone.

    uv run --extra explore experiments/triple/transfer.py     -> docs/triple/03_transfer.md
"""

import pickle

import pandas as pd
from triple import DOCS, PROC, config, equations, md, physics, rates, score, table, with_missing_types

from sindy_wm.models.intent_model import add_controls
from sindy_wm.models.pipeline import Layout, Pipeline

COLS = ["health_1s", "health_2s", "health_4s", "margin_r2_2s", "margin_r2_4s", "alive_4s", "distance_2s"]


def main() -> None:
    triple = pickle.load(open(PROC / "exp12_triple_typed" / "triple_model.pkl", "rb"))
    frozen = pickle.load(open(PROC / "final_exploration" / "generalisation_frozen.pkl", "rb"))
    stalk = Pipeline(("STALKER",), withdraw_radius=7.0 * 7.25 / 5.75, **config()).fit(
        rates("exp07_stalkers", "train", Layout(("stalker",)))
    )
    pickle.dump(stalk, open(PROC / "exp07_stalkers_typed" / "stalker_model.pkl", "wb"))
    lines = ["# Exploratory: stalkers alone, and one model for several scenarios (val only)", ""]
    lines += [
        "## Stalkers alone (exp07 train 1500, val 300; reach capped at 6)",
        "",
        "```",
        equations(stalk),
        "```",
        "",
    ]
    lines += [md(physics(stalk).round(3)), ""]
    rows = []
    for name, exp, own in (
        ("marines", "exp06_intents", frozen["marines"]),
        ("marauders", "exp09_marauders", frozen["marauders"]),
        ("stalkers", "exp07_stalkers", stalk),
        ("marines + marauders", "exp08_mixed", frozen["mixed"]),
    ):
        st = add_controls(table(exp, "val"))
        scale = own.L.hp[own.L.types[0]] / triple.L.hp["marine"]  # own health unit, in marines' hp
        t_own, _, (h_own, _), _ = score(own, st, {"persistence": None})
        t_tri, _, (h_tri, _), _ = score(triple, add_controls(with_missing_types(st, triple.L.types)), {})
        for label, tab, h, f in (
            ("persistence", t_own.loc["persistence"], None, scale),
            ("own model", t_own.loc["pipeline"], h_own, scale),
            ("triple model, unchanged", t_tri.loc["pipeline"], h_tri, 1.0),
        ):
            r = tab[COLS].copy()
            r[[c for c in COLS if c.startswith("health")]] *= f
            rows.append({"val set": name, "model": label, **r.to_dict(), "horizon": h})
    lines += [
        "## One model, several scenarios (health in ten marines' hp)",
        "",
        md(pd.DataFrame(rows).round(4)),
        "",
    ]
    (DOCS / "03_transfer.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
