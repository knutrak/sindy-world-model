# ruff: noqa: E402, E501, E702, E731, B007, B023, I001
"""Stage 2: fit the finalists and B1 on all prepared train rows, and on marines and marine variants only (LO)."""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit as F

ROOT = Path(__file__).resolve().parents[2]
B = {"share": "scaled", "tau": "scaled", "tau_s": 2.0}
OPTS = {
    "B1": {},
    "H5": {**B, "form": 1, "thr_form": 0.2, "over": "both", "vd": 0.125},
    "I7": {**B, "over": "sat", "vd": 0.125},
}
ALL = ("marines", "marauders", "stalkers", "mixed", "triple", *F.VARIANTS)
LO = ("marines", *F.VARIANTS)

if __name__ == "__main__":
    out = {(n, w): F.fit(F.pool_of(p), o) for n, o in OPTS.items() for w, p in (("all", ALL), ("LO", LO))}
    pickle.dump(out, open(F.CACHE / "finalists.pkl", "wb"))
    plain = lambda v: (
        {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, dict)
        else (v.tolist() if hasattr(v, "tolist") else (list(v) if isinstance(v, tuple) else v))
    )  # noqa: E731
    (ROOT / "docs" / "overnight" / "finalists.json").write_text(
        json.dumps({f"{n} {w}": plain(p) for (n, w), p in out.items()}, indent=1)
    )
    for (n, w), p in out.items():
        print(
            n,
            w,
            {k: round(v, 4) for k, v in p["eta"].items()},
            p["over"],
            {s: {k: round(x, 3) for k, x in v.items()} for s, v in p["p"].items()},
            dict(zip(p["dist_names"], p["dist_coef"].round(3), strict=True)),
            {f"{a}-{b}": {k: round(x, 3) for k, x in v.items()} for (a, b), v in p.get("form", {}).items()},
            "terms",
            F.complexity(p),
        )
