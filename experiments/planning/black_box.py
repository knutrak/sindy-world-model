"""Black-box world model in the same planner (approved plan: docs/planning/36_black_box_planner_plan.md).

uv run --extra explore experiments/planning/black_box.py time       2 epochs on mixed, timed
uv run --extra explore experiments/planning/black_box.py train      both models -> mlp_models.pkl
uv run --extra explore experiments/planning/black_box.py b0         gate B0 and val errors
uv run experiments/planning/closed34.py run S2 --n 400 --mlp        closed loop (policy mlp:h3)
uv run --extra explore experiments/planning/black_box.py analyse    -> docs/planning/37_black_box_planner.md
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "generalisation"))
import closed34 as C34  # noqa: E402
import mu_sigma as MS  # noqa: E402

from sindy_wm.models.intent_model import DT  # noqa: E402
from sindy_wm.models.kite import CTRL_K  # noqa: E402
from sindy_wm.models.mu_sigma import attach_margins  # noqa: E402
from sindy_wm.models.reach_state import ReachStatePipeline  # noqa: E402
from sindy_wm.models.typed_mlp import TypedMLPWorld  # noqa: E402
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

OUT = DATA_DIR / "processed" / "planning"
MODELS = OUT / "mlp_models.pkl"
RES = OUT / "black_box.json"
DOC = PROJECT_ROOT / "docs" / "planning" / "37_black_box_planner.md"
N_BOOT = 2000


def train_batch(scen):
    m = MS.frozen(scen)
    st = MS.load(scen, "train")
    with np.errstate(invalid="ignore", divide="ignore"):
        B = attach_margins(m.L.batch(st, horizon=24, every=8, history=8, ctrl=CTRL_K), st, m.L)
    return m, B


def time_it():
    m, B = train_batch("mixed")
    t0 = time.time()
    TypedMLPWorld(m.L, m.max_reach, epochs=2).fit(B)
    print(f"2 epochs: {time.time() - t0:.0f} s; starts available {len(B['X0'])}")


def train():
    import torch

    torch.set_num_threads(4)
    models = {}
    for scen in MS.SCEN:
        m, B = train_batch(scen)
        t0 = time.time()
        models[scen] = TypedMLPWorld(m.L, m.max_reach).fit(
            B, log=lambda s, scen=scen: print(scen, s, flush=True)
        )
        print(f"{scen}: {time.time() - t0:.0f} s, {models[scen].n_params} weights", flush=True)
    pickle.dump(models, open(MODELS, "wb"))


def b0():
    mlp = pickle.load(open(MODELS, "rb"))
    ms = pickle.load(open(MS.MODELS_OUT, "rb"))
    rng = np.random.default_rng(0)
    rows, vrows = [], []
    ok = True
    for scen in MS.SCEN:
        m = MS.frozen(scen)
        models = {"frozen": ReachStatePipeline.from_frozen(m, None), "mlp": mlp[scen], "mu-sigma": ms[scen]}
        names = [n for _, n in MS.SCEN[scen]["cf"]]
        st = MS.from_branch(MS.load(scen, "cf", branch=True), m.L, len(names))
        R, _ = MS.regrets(models, st, m, names)
        for h, reg in R.items():
            row = {"scenario": scen, "h": h, **{f"regret {n}": reg[n].mean() for n in reg}}
            for a, b in (("mlp", "frozen"), ("mlp", "mu-sigma")):
                d = reg[a] - reg[b]
                mm = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
                lo, hi = np.quantile(mm, [0.025, 0.975])
                row[f"{a} − {b}"] = f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
                if b == "frozen":
                    row["below frozen"] = bool(hi < 0)
            rows.append(row)
        ok &= all(r["below frozen"] for r in rows if r["scenario"] == scen and r["h"] in (2.0, 4.0))
        # val errors
        sv = MS.load(scen, "val")
        hz = 32
        with np.errstate(invalid="ignore", divide="ignore"):
            B = attach_margins(m.L.batch(sv, horizon=hz, ctrl=CTRL_K), sv, m.L)
        TY = B["team"]["Y"]
        for name, mod in models.items():
            with np.errstate(all="ignore"):
                Y = m.L.team_view(mod.rollout(B))
            for h in (1.0, 2.0, 4.0):
                n = round(h / DT)
                okr = ~np.isnan(TY[:, n, 0])
                real = TY[okr, n, 0] - TY[okr, n, 1] - (TY[okr, 0, 0] - TY[okr, 0, 1])
                pred = Y[okr, n, 0] - Y[okr, n, 1] - (Y[okr, 0, 0] - Y[okr, 0, 1])
                vrows.append(
                    {
                        "scenario": scen,
                        "model": name,
                        "h": h,
                        "health error": np.abs(Y[okr, n, :2] - TY[okr, n, :2]).sum(1).mean(),
                        "margin error": np.abs(pred - real).mean(),
                        "margin R²": 1 - ((pred - real) ** 2).sum() / ((real - real.mean()) ** 2).sum(),
                    }
                )
        print(scen, "done", flush=True)
    json.dump({"b0": rows, "val": vrows, "passes": bool(ok)}, open(RES, "w"), indent=1, default=float)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(vrows).round(4).to_string(index=False))
    print("B0 passes:", ok)


def analyse():
    rng = np.random.default_rng(0)
    rows, diag, wins = [], [], []
    for scen in C34.FIRST:
        d = pd.read_csv(C34.A.OUT / f"closed34_{scen}.csv")
        if "mlp:h3" not in set(d.policy):
            continue
        wins.append(d.groupby("policy")["won"].mean().rename(scen))
        comps = [("D1" if scen in C34.MAIN else "D3", C34.BEST_KITE[scen]), ("D2", "musigma:h3")]
        if scen not in C34.MAIN:
            comps.append(("D3", C34.BEST_FIXED[scen]))
        for label, ref in comps:
            mean, lo, hi, read, n = C34.paired(d, "mlp:h3", ref, rng)
            rows.append(
                {
                    "scenario": scen,
                    "rule": label,
                    "against": ref.removeprefix("fixed:"),
                    "difference": mean,
                    "95%": f"[{lo:+.3f}, {hi:+.3f}]",
                    "read-out": read,
                    "n": n,
                }
            )
        p = d[d.policy == "mlp:h3"]
        cols = [c for c in p if c.startswith("pick_")] + ["near_tie_share", "switches", "wall_s"]
        diag.append({"scenario": scen, **p[[c for c in cols if c in p]].mean().to_dict()})
    t = pd.DataFrame(rows)
    d1 = t[t.rule == "D1"]["read-out"]
    if (d1 == "beats").any() and not (d1 == "loses").any():
        verdict = (
            "the interpretable model limits planning (D1 beats in at least one main scenario, loses in none)"
        )
    elif (d1 == "loses").all():
        verdict = "the setup limits planning (D1 loses in every main scenario)"
    else:
        verdict = "in between: reported per scenario, no single statement"
    b = json.load(open(RES))
    pd.set_option("display.width", 250)
    lines = [
        "# Black-box world model in the same planner: results",
        "",
        "Plan: `36_black_box_planner_plan.md` (approved 22 Nov).",
        "Script: `experiments/planning/black_box.py`.",
        "",
        f"## Gate B0 (counterfactual ranking): {'passes' if b['passes'] else 'fails'}",
        "",
        "```",
        pd.DataFrame(b["b0"]).round(4).to_string(index=False),
        "```",
        "",
        "## Val errors",
        "",
        "```",
        pd.DataFrame(b["val"]).round(4).to_string(index=False),
        "```",
        "",
        "## Closed loop: wins per policy (34's seeds)",
        "",
        "```",
        pd.concat(wins, axis=1).round(3).to_string(),
        "```",
        "",
        "## Paired comparisons (MLP planner − reference)",
        "",
        "```",
        t.round(3).to_string(index=False),
        "```",
        "",
        f"**Interpretation fixed in the plan: {verdict}.**",
        "",
        "## Planner diagnostics",
        "",
        "```",
        pd.DataFrame(diag).round(3).to_string(index=False),
        "```",
        "",
    ]
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["time", "train", "b0", "analyse"])
    {"time": time_it, "train": train, "b0": b0, "analyse": analyse}[p.parse_args().what]()


if __name__ == "__main__":
    main()
