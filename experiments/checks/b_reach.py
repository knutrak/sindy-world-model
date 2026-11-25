# ruff: noqa: B023, E402, E501, UP031
"""Why model B lowers firepower: A and B with their own or the measured share in reach (val). docs/checks/README.md"""

import pickle
import sys
import warnings

sys.path.insert(0, "notebooks/lib")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sindyc_07_lab as lab

B = lab.load_batch("val")
P = pickle.load(open("data/processed/exp06_intents/sindyc_07_models.pkl", "rb"))
A, Bm, S = P["A model_derived"], P["B refit, same terms"], P["S share as a state"]
print(len(B["X0"]), "starts", {k: getattr(v, "shape", None) for k, v in B.items()})
# true share over the steps 0..T-1
Y = B["Y"]
T = B["U"].shape[1]
Yr = B["Yreach"]
print("Y", Y.shape, "Yreach", Yr.shape)


def carry(M):
    M = M.copy()
    for n in range(1, M.shape[1]):
        bad = np.isnan(M[:, n])
        M[:, n][bad] = M[:, n - 1][bad]
    return M


Yc, Rc = carry(Y), carry(Yr)
# Yreach[:, n-1] is compared with R used in step n-1 (see residuals) -> index n = state at step n
al = Yc[:, : Rc.shape[1], 2:4]
share = np.clip(np.divide(Rc, np.maximum(al, 1e-9)), 0, 1)
truth = {"share": share[:, :T]}
KB = dict(A, kb=Bm["kb"], kr=Bm["kr"], kr_wd=Bm["kr_wd"])
BK = dict(Bm, kb=A["kb"], kr=A["kr"], kr_wd=A["kr_wd"])
models = {"A": A, "A with B's k": KB, "B with A's k": BK, "B": Bm, "S": S}
rows = {}
for name, p in models.items():
    for lab_, t in (("own closure", None), ("measured share", truth)):
        if name == "S" and t is not None:
            continue
        rows[(name, lab_)] = lab.metrics(lab.simulate(p, B, truth=t), B)
d = pd.DataFrame(rows).T
print(d[[c for c in d.columns if "health" in c or "margin" in c]].round(3).to_string())
print("\nsigned bias in units in reach (model - measured), x10 units, and in shares; by horizon")
for name, p in (("A", A), ("B", Bm), ("S", S)):
    y, R = lab.simulate(p, B, return_reach=True)
    for h in (1, 2, 4):
        n = int(h * 8)
        e = 10 * (R[:, n - 1] - Yr[:, n - 1])
        print(
            name,
            h,
            "blue {:+.2f} red {:+.2f}".format(*tuple(np.nanmean(e, 0))),
            "| signed loss error (model-true health) blue {:+.4f} red {:+.4f}".format(
                *tuple(np.nanmean(y[:, n, :2] - Y[:, n, :2], 0))
            ),
        )
# what k would turn the model's own predicted reach into the true damage over 1..4 s (least squares through 0)
y, R = lab.simulate(A, B, return_reach=True)
for h in (1, 2, 4):
    n = 8 * h
    ok = ~np.isnan(Y[:, n, 0])
    Iu = R[ok][:, :n].sum(1) * lab.DT
    It = Rc[ok][:, :n].sum(1) * lab.DT
    for j, o, kk in ((0, 1, "Blue loss"), (1, 0, "Red loss")):
        loss = B["X0"][ok, j] - Y[ok, n, j]
        nw = (B["U"][ok, :n, 2].max(1) == 0) if j == 1 else np.ones(ok.sum(), bool)

        def f(x):
            return (x[nw] * loss[nw]).sum() / (x[nw] ** 2).sum()

        print(
            h,
            kk,
            f"k on measured reach {f(It[:, o]):.3f}, on A's predicted reach {f(Iu[:, o]):.3f}",
            f"corr {np.corrcoef(It[nw, o], loss[nw])[0, 1]:.3f} {np.corrcoef(Iu[nw, o], loss[nw])[0, 1]:.3f}",
        )
