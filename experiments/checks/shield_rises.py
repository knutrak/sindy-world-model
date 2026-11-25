# ruff: noqa: B023, E402, E501, UP031
"""When does stalker health rise (exp07 train)?"""

import pandas as pd

from sindy_wm.models.intent_model import DT
from sindy_wm.paths import DATA_DIR

f = DATA_DIR / "processed" / "exp07_stalkers_typed" / "train.parquet"
d = pd.read_parquet(f)
print(len(d), [c for c in d if "stalker" in c][:20])
for s in ("blue", "red"):
    o = "red" if s == "blue" else "blue"
    h = 1600 * d[f"health_{s}_stalker"]
    g = h.groupby(level=0).diff().shift(-1)
    ep = d.index.get_level_values(0)
    ok = (g.notna()) & (d[f"alive_{s}_stalker"] > 0)
    R = 10 * d[f"reach_{o}_stalker"]
    x = g[ok]
    r = R[ok]
    contacted = (R > 0).groupby(level=0).cumsum().gt(0)[ok]
    for lab, m in (
        ("before first contact", ~contacted),
        ("in contact, enemy in reach", (r > 0) & contacted),
        ("later, no enemy in reach", (r == 0) & contacted),
    ):
        v = x[m]
        print(
            f"{s} {lab}: {m.mean():.2f} of steps; rises in {(v > 0).mean():.2f}; gained {v[v > 0].sum() / len(x):.2f} hp per step, lost {-v[v < 0].sum() / len(x):.2f}"
        )
    print(
        "   total regained / lost: %.3f" % (x[x > 0].sum() / -x[x < 0].sum()),
        "| mean gain when rising hp/s %.2f" % (x[x > 0].mean() / DT),
    )
