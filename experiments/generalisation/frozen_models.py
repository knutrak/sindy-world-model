"""Fit the frozen pipeline (configs/generalisation_pipeline.yaml) on each scenario's train set, save the
models and write their equations (docs/generalisation/06_frozen_equations.md). Reads train only.

    uv run --extra explore experiments/generalisation/frozen_models.py
"""

import pickle
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DOCS, rows  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.models.pipeline import Pipeline  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

CONFIG = PROJECT_ROOT / "configs" / "generalisation_pipeline.yaml"
OUT = H.CACHE / "generalisation_frozen.pkl"
SPEC = ("withdraw", "alloc", "thr_health", "thr_distance", "max_reach", "movement_wd")


def fit_all() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text())
    kw = {k: cfg["pipeline"][k] for k in SPEC}
    models = {}
    for name in ("marines", "marauders", "mixed"):
        s = cfg["scenarios"][name]
        models[name] = Pipeline(tuple(s["types"]), withdraw_radius=s["withdraw_radius"], **kw).fit(rows(name))
    return models


def equations(m: Pipeline) -> str:
    """The model as readable lines (health, alive, distance)."""
    side = {"blue": "B", "red": "R"}
    lines = []
    for (s, t), c in m.coef.items():
        terms = " ".join(f"{v:+.4f}·{k}" for k, v in c.items() if v)
        lines.append(f"H_{side[s]}^{t}' = {terms}")
    for (s, t), (p, pf) in m.p.items():
        f = f" {pf:+.3f}·focus" if pf else ""
        lines.append(f"(ln A_{side[s]}^{t})' = ({p:.3f}{f}) · (ln H_{side[s]}^{t})'")
    d = " ".join(f"{c:+.3f}·{n}" for n, c in zip(m.dist_names, m.dist_coef, strict=True) if c)
    lines.append(f"d' = {d}")
    return "\n".join(lines)


def main() -> None:
    models = fit_all()
    pickle.dump(models, open(OUT, "wb"))
    out = [
        "# The frozen pipeline's models, fitted on each scenario's train set (21 Nov)",
        "",
        "Script: `experiments/generalisation/frozen_models.py`;",
        "config: `configs/generalisation_pipeline.yaml`.",
        "R terms: enemy units in reach (×(1 − wd) for Blue attackers), s[t]: share of fire on type t",
        "(proportional rule). danger·wd, walk_r·wd: the withdraw movement rule.",
        "Units: target units per second.",
        "",
    ]
    for name, m in models.items():
        out += [
            f"## {name}",
            "",
            "```",
            equations(m),
            "```",
            "",
            f"reach closure rho: {m.rho}; complexity {m.complexity()}",
            "",
        ]
    (DOCS / "06_frozen_equations.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
