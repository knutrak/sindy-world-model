"""Moment-closure feasibility check (docs/generalisation/12_moment_closure_check.md; exploratory,
criterion fixed before the run): does a 7-group physics library explain μ′ and σ′ about as well as the
dense one, with walking coefficients near the game's speed?

    uv run --extra explore experiments/generalisation/moment_closure.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mu_sigma import OUT, SCEN, controls_for, frozen, load  # noqa: E402

from sindy_wm.models.mu_sigma import group_names, rows_for  # noqa: E402
from sindy_wm.models.pipeline import SIDES  # noqa: E402
from sindy_wm.models.reach_state import gram, group_index, two_level_gram  # noqa: E402
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

DOC = PROJECT_ROOT / "docs" / "generalisation" / "12_moment_closure_check.md"
PHYS = ["1", "q", "q_o", "phi", "phi_o", "dA", "dAo"]
V = 3.15
THR = 0.05


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main() -> None:
    ck = pickle.load(open(OUT / "fits.pkl", "rb"))["ck"]
    rows, coefs = [], []
    for scen in SCEN:
        m = frozen(scen)
        tt = m.L.types
        tr, va = load(scen, "train"), load(scen, "val")
        ctrls = controls_for(scen)
        nc = len(ctrls)
        for side in SIDES:
            for a in tt:
                for target in ("mu", "sd"):
                    out = {"scenario": scen, "equation": f"{target}′ {side} {a}"}
                    for lib, names in (("physics", PHYS), ("dense", group_names(True))):
                        for meth, rad in (("step", 8), ("weak", 8)):
                            kw = dict(radius=rad, names=names)
                            Th, b, _, nm = rows_for(
                                tr, tt, side, a, target, True, ctrls, meth, *ck[scen], **kw
                            )
                            g, base = group_index(nm, ctrls)
                            c = two_level_gram(*gram(Th, b), g, base, THR)
                            Tv, bv, _, _ = rows_for(
                                va, tt, side, a, target, True, ctrls, meth, *ck[scen], **kw
                            )
                            out[f"{lib} {meth} val R²"] = r2(bv, Tv.astype(float) @ c)
                            if meth == "step":
                                out[f"{lib} groups"] = len(np.unique(g[c != 0]))
                            if lib == "physics" and meth == "step":
                                cf = {f"{nm[j // nc]}·{ctrls[j % nc]}": v for j, v in enumerate(c) if v}
                                coefs.append({"scenario": scen, "equation": out["equation"], **cf})
                                if target == "mu":
                                    eff = {
                                        gname: c[nm.index(gname) * nc] + c[nm.index(gname) * nc + 1]
                                        for gname in ("q", "q_o")
                                    }
                                    out["eff q (1+adv)"], out["eff q_o (1+adv)"] = eff["q"], eff["q_o"]
                                    out["physical"] = bool(
                                        all(-1.5 * V <= e <= -0.5 * V for e in eff.values())
                                    )
                    rows.append(out)
                    print(out, flush=True)
    t = pd.DataFrame(rows)
    ratio = float((t["physics step val R²"] / t["dense step val R²"]).median())
    n_phys = int(t["physical"].fillna(False).sum())
    passed = ratio >= 0.8 and n_phys >= 4
    pd.set_option("display.width", 250)
    lines = [
        "",
        "**Run 22 Nov.** Val R² of μ′ and σ′ (one-step SG-9 rates; weak form radius 1 s alongside):",
        "",
        "```",
        t.round(3).to_string(index=False),
        "```",
        "",
        f"- Median physics / dense val R² (one-step): **{ratio:.2f}** (criterion ≥ 0.80).",
        f"- μ′ equations with both walking coefficients in [−4.7, −1.6]: **{n_phys} of 6** (criterion ≥ 4).",
        f"- **Criterion: {'passes' if passed else 'fails'}.**",
        "",
        "Physics coefficients (one-step, threshold 0.05; name·control):",
        "",
        "```",
        pd.DataFrame(coefs).round(3).T.to_string(),
        "```",
        "",
    ]
    DOC.write_text(DOC.read_text().replace("(filled in after the run)\n", "\n".join(lines)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
