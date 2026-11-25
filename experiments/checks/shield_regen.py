# ruff: noqa: B023, E402, E501, UP031
"""Can a regeneration term be found in stalker health (exp07 train)?"""

import numpy as np
import pandas as pd

from sindy_wm.models.intent_model import r2, stlsq
from sindy_wm.paths import DATA_DIR

r = pd.read_parquet(DATA_DIR / "processed" / "exp07_stalkers_typed" / "rates_train_w9.parquet")
for s in ("blue", "red"):
    o = "red" if s == "blue" else "blue"
    y = r[f"d_health_{s}_stalker"].to_numpy()
    R = r[f"reach_{o}_stalker"].to_numpy().copy()
    if s == "red":
        R = R * (1 - r["withdraw"].to_numpy())
    Aown = r[f"alive_{s}_stalker"].to_numpy()
    Aopp = r[f"alive_{o}_stalker"].to_numpy()
    Hdef = Aown - r[f"health_{s}_stalker"].to_numpy()
    one = np.ones(len(y))
    print(
        f"\n== {s} loses; health unit = 10 stalkers (1600 hp); regen if every unit regenerates: +{2 * 10 / 1600:.4f} per unit of A (tens)"
    )
    for lab, cols in (
        ("R only", {"R": R}),
        ("R + own alive", {"R": R, "A_own": Aown}),
        ("R + own missing health", {"R": R, "missing": Hdef}),
        ("R + opponent alive (in library)", {"R": R, "A_opp": Aopp}),
        ("R + constant", {"R": R, "1": one}),
    ):
        X = np.column_stack(list(cols.values()))
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        es = [abs(v) * (x.std() if x.std() > 1e-9 else 1) / y.std() for v, x in zip(c, X.T, strict=False)]
        print(
            f"  {lab:32s}",
            {k: round(float(v), 4) for k, v in zip(cols, c, strict=False)},
            "effect",
            [round(float(e), 3) for e in es],
            f"R2 {r2(y, X @ c):.4f}",
            "k/game %.2f" % (-c[0] / 0.079),
        )
    Xall = np.column_stack([R, Aown, Hdef, Aopp, one])
    print(
        "  STLSQ 0.15 / 0.05 / 0.02 on all:",
        [
            dict(zip(["R", "A_own", "missing", "A_opp", "1"], np.round(stlsq(Xall, y, t), 4), strict=False))
            for t in (0.15, 0.05, 0.02)
        ],
    )
