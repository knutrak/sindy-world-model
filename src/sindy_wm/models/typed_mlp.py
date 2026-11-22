"""A black-box world model on the pipeline's typed state (docs/planning/36_black_box_planner_plan.md).

x_{t+1} = x_t + DT * f(x_t, c, u_t), with x the typed state (health and units alive per side and type,
distance), c constant inputs from the start point (shape, front offsets, start share in reach, start μ and σ
of the reach margins) and u the five controls (CTRL + kite). f: 2 hidden layers of 128, tanh. Trained on
rollout loss over the planner's horizon (as `final_exploration`'s MLPWorld). Outputs clipped: >= 0, health
and alive never above their start. Same interface as the pipeline for the planner: `L`, `max_reach`,
`rollout(B)`, `control_vector`.
"""

from __future__ import annotations

import numpy as np

from sindy_wm.models.intent_model import DT
from sindy_wm.models.kite import control_vector_k


class TypedMLPWorld:
    control_vector = staticmethod(control_vector_k)
    needs_margins = True  # the planner attaches the live μ, σ (docs/planning/34)

    def __init__(
        self,
        layout,
        max_reach: float,
        hidden: int = 128,
        steps: int = 24,
        epochs: int = 30,
        n_start: int = 40_000,
        seed: int = 0,
    ):
        self.L, self.max_reach = layout, max_reach
        self.hidden, self.steps, self.epochs, self.n_start, self.seed = hidden, steps, epochs, n_start, seed

    # ---- inputs
    def _const(self, B) -> np.ndarray:
        n = self.L.n
        A0 = B["X0"][:, 2 * n : 4 * n]
        share0 = np.divide(B["reach0"], A0, out=np.zeros_like(A0), where=A0 > 1e-9)
        return np.column_stack(
            [
                np.nan_to_num(B["shape0"]),
                np.nan_to_num(B["front0"]),
                share0,
                np.nan_to_num(B["mu0"]),
                np.nan_to_num(B["sd0"]),
            ]
        ).astype(np.float32)

    def _scale(self):
        import torch

        n = self.L.n
        return torch.tensor([0.05] * (2 * n) + [0.1] * (2 * n) + [0.5], dtype=torch.float32)

    def _roll(self, X0, C, U, steps):
        import torch

        X, out = X0, [X0]
        for t in range(steps):
            z = (torch.cat([X, C, U[:, t]], -1) - self.mu) / self.sd
            Xn = torch.clamp(X + DT * self.net(z), min=0)
            k = X0.shape[1] - 1
            Xn = torch.cat([torch.minimum(Xn[:, :k], X0[:, :k]), Xn[:, k:]], -1)
            X = Xn
            out.append(X)
        return torch.stack(out, 1)

    # ---- fit
    def fit(self, B: dict, log=print) -> TypedMLPWorld:
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(B["X0"]), min(self.n_start, len(B["X0"])), replace=False)
        X0 = torch.tensor(B["X0"][idx], dtype=torch.float32)
        C = torch.tensor(self._const(B)[idx])
        U = torch.tensor(B["U"][idx, : self.steps], dtype=torch.float32)
        Y = torch.tensor(B["Y"][idx, : self.steps + 1], dtype=torch.float32)
        mask = ~torch.isnan(Y[..., 0])
        Y = torch.nan_to_num(Y)
        inp = torch.cat([X0, C, U[:, 0]], -1)
        self.mu, self.sd = inp.mean(0), inp.std(0) + 1e-3
        scale = self._scale()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(inp.shape[1], self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, X0.shape[1]),
        )
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        n = len(X0)
        for ep in range(self.epochs):
            perm, tot = torch.randperm(n), 0.0
            for k in range(0, n, 512):
                i = perm[k : k + 512]
                pred = self._roll(X0[i], C[i], U[i], self.steps)
                err = ((pred - Y[i]) / scale) ** 2
                loss = (err * mask[i][..., None]).sum() / mask[i].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * len(i)
            sched.step()
            log(f"epoch {ep + 1}/{self.epochs}: loss {tot / n:.4f}")
        self.n_params = sum(p.numel() for p in self.net.parameters())
        return self

    # ---- rollout
    def rollout(self, B: dict) -> np.ndarray:
        import torch

        with torch.no_grad():
            X0 = torch.tensor(np.nan_to_num(B["X0"]), dtype=torch.float32)
            C = torch.tensor(self._const(B))
            U = torch.tensor(B["U"], dtype=torch.float32)
            Y = self._roll(X0, C, U, U.shape[1]).numpy().astype(float)
        k = B["X0"].shape[1] - 1
        Y[:, :, :k] = np.minimum(Y[:, :, :k], np.nan_to_num(B["X0"])[:, None, :k])
        Y[:, 0] = B["X0"]
        return Y
