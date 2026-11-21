"""Check: does sharing SMAClite's unit-order shuffle stream give the perturbed oracle extra foresight?

Compares the perturbed oracle (copies keep the real game's shuffle stream) with the same oracle whose
copies get their own stream (`reseed=True`), paired on the same seeds, and both against the best fixed
intent. Reads headroom_<S>.csv and headroom_<S>_reseed.csv.

    uv run experiments/planning/reseed_check.py
"""

import numpy as np
import pandas as pd

from sindy_wm.paths import DATA_DIR

IN = DATA_DIR / "processed" / "planning"
OBJ = ["won", "margin", "log_exchange"]


def ci(d: np.ndarray, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return f"{d.mean():+.3f} [{np.quantile(b, 0.025):+.3f}, {np.quantile(b, 0.975):+.3f}]"


def main() -> None:
    for path in sorted(IN.glob("headroom_S*_reseed.csv")):
        scen = path.stem.split("_")[1]
        t = pd.concat([pd.read_csv(IN / f"headroom_{scen}.csv"), pd.read_csv(path)])
        t["won"] = t["won"].astype(float)
        t["log_exchange"] = np.log(t["exchange"])
        seeds = sorted(set(t.loc[t.policy == "oracle-perturbed-reseed", "seed"]))
        fixed = t[t.policy.str.startswith("fixed:")].groupby("policy")["won"].mean()
        best = fixed.idxmax()
        piv = {
            p: t[t.policy == p].set_index("seed").loc[seeds]
            for p in ("oracle", "oracle-perturbed", "oracle-perturbed-reseed", best)
        }
        print(f"\n{scen} ({len(seeds)} paired seeds; best fixed on win: {best})")
        rows = {}
        for o in OBJ:
            a, b, r, f = (
                piv[p][o].to_numpy() for p in ("oracle", "oracle-perturbed", "oracle-perturbed-reseed", best)
            )
            rows[o] = {
                "exact": a.mean(),
                "perturbed": b.mean(),
                "perturbed+reseed": r.mean(),
                "best fixed": f.mean(),
                "reseed − perturbed": ci(r - b),
                "reseed − best fixed": ci(r - f),
            }
        print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.3f}"))
        same = (piv["oracle-perturbed"]["won"] == piv["oracle-perturbed-reseed"]["won"]).mean()
        print(f"same winner as perturbed: {same:.2f}")


if __name__ == "__main__":
    main()
