# ruff: noqa: E402, E501, E702, E731, B007, B023, I001
"""Stage 0, B0 and B2: each army's own fit, and a black box with the same inputs, on the same score battles."""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit as F
import numpy as np
import pandas as pd
import search as Se
import torch

from sindy_wm.models.param import rate, shots
from sindy_wm.states.typed import unit_stats

torch.set_num_threads(4)
SINGLE = ("marines", "marauders", "stalkers", *F.VARIANTS)


def props(arm):
    t = F.army(arm)["types"][0]
    s = unit_stats(t)
    _, _, m = F.geometry((t,))
    return np.array([rate(t, t), 7.5 / shots(t, t), 5.75 / m.rho[t.lower()], s["speed"] / 3.15])


def inputs(arm, B):
    d = F.army(arm)
    ds, vs, _ = F.geometry(d["types"])
    T = B["team"]
    C = np.column_stack(
        [
            np.nan_to_num(T["shape0"]) * ds,
            T["reach0"],
            np.nan_to_num(B["mu0"]) * ds,
            np.nan_to_num(B["sd0"]) * ds,
            B["vd0"],
            np.repeat(props(arm)[None], len(T["X0"]), 0),
        ]
    )
    return T["X0"] * [1, 1, 1, 1, ds / 5.75], C, T["U"], T["Y"] * [1, 1, 1, 1, ds / 5.75]


def train(names, steps=16, epochs=25):
    torch.manual_seed(0)
    P = [inputs(a, F.army(a)["B"]) for a in names]
    X0, C, U, Y = (
        torch.tensor(
            np.concatenate(
                [p[i][:, : steps + 1] if i == 3 else (p[i][:, :steps] if i == 2 else p[i]) for p in P]
            ),
            dtype=torch.float32,
        )
        for i in range(4)
    )
    ok = ~torch.isnan(Y[..., 0])
    Y = torch.nan_to_num(Y)
    inp = torch.cat([X0, C, U[:, 0]], -1)
    mu, sd = inp.mean(0), inp.std(0) + 1e-2
    net = torch.nn.Sequential(
        torch.nn.Linear(inp.shape[1], 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 5),
    )
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sc = torch.tensor([0.05, 0.05, 0.1, 0.1, 0.1])

    def roll(x, c, u):
        out = [x]
        for t in range(u.shape[1]):
            xn = torch.clamp(out[-1] + 0.125 * net((torch.cat([out[-1], c, u[:, t]], -1) - mu) / sd), min=0)
            out.append(torch.cat([torch.minimum(xn[:, :4], x[:, :4]), xn[:, 4:]], -1))
        return torch.stack(out, 1)

    for _ in range(epochs):
        for i in torch.randperm(len(X0)).split(256):
            e = (((roll(X0[i], C[i], U[i]) - Y[i]) / sc) ** 2 * ok[i][..., None]).mean()
            opt.zero_grad()
            e.backward()
            opt.step()
    return roll


def main():
    base = pickle.load(open(Se.OUT / "B1.pkl", "rb"))["res"]
    rows = []
    for fold, left in (
        ("marauders", "marauders"),
        ("stalkers", "stalkers"),
        ("range", "R4"),
        ("range", "R3"),
        ("deadly", "DP"),
        ("deadly", "CM"),
        ("deadly", "HM"),
        ("deadly", "SP"),
    ):
        out = [a for a in SINGLE if a not in Se.FOLDS[fold][1]]
        roll = train(out)
        X0, C, U, _ = inputs(left, F.army(left)["B"])
        T = F.army(left)["B"]["team"]
        with torch.no_grad():
            Y = roll(*(torch.tensor(v, dtype=torch.float32) for v in (X0, C, U[:, :16]))).numpy()
        bb = float(np.nanmean(np.abs(Y[:, 16, :2] - T["Y"][:, 16, :2])))
        own = Se.errors(F.model(F.fit({left: 40_000}, {}), left), F.army(left)["B"])
        rows.append(
            {
                "army": left,
                "B1 (left out)": np.nanmean(base[fold][left]["e2"]),
                "B0 own fit": np.nanmean(own["e2"]),
                "B2 black box (left out)": bb,
            }
        )
    t = pd.DataFrame(rows)
    t.to_csv(Se.OUT / "ceiling.csv", index=False)
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
