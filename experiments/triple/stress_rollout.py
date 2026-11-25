# ruff: noqa: E501, E731, B007, B023, B905, I001
"""Stress tests of the rollouts, the state builder and the world-model interface (docs/triple/04_stress.md).

uv run --extra explore experiments/triple/stress_rollout.py
"""

import copy
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0]]
from stress_core import head  # noqa: E402
from triple import PROC  # noqa: E402

from sindy_wm.models.pipeline import SIDES  # noqa: E402
from sindy_wm.states.typed import typed_states  # noqa: E402
from sindy_wm.world_model import INTENTS, WorldModel, plan_controls  # noqa: E402

frozen = pickle.load(open(PROC / "final_exploration" / "generalisation_frozen.pkl", "rb"))
M = dict(frozen)
M["triple"] = pickle.load(open(PROC / "exp12_triple_typed" / "triple_model.pkl", "rb"))
M["stalkers"] = pickle.load(open(PROC / "exp07_stalkers_typed" / "stalker_model.pkl", "rb"))
rng = np.random.default_rng(0)


def batch(m, n, steps, lo=0.0, hi=1.5, dmin=0.0, dmax=16.0, zero=0.2):
    k = 2 * m.L.n
    A = rng.uniform(lo, hi, (n, k)) / m.L.n
    H = A * rng.uniform(0.02, 1, (n, k))
    z = rng.random((n, k)) < zero
    A[z] = H[z] = 0
    U = np.stack([plan_controls(INTENTS[i % 5], steps * 0.125) for i in range(n)])
    return {
        "X0": np.column_stack([H, A, rng.uniform(dmin, dmax, n)]),
        "shape0": rng.uniform(0, 3, (n, 4)),
        "front0": rng.uniform(-2, 2, (n, k)),
        "reach0": A * rng.uniform(0, 1, A.shape),
        "U": U,
    }


def check(m, B, Y):
    n = m.L.n
    H, A, d = Y[..., : 2 * n], Y[..., 2 * n : 4 * n], Y[..., -1]
    return {
        "non-finite": int((~np.isfinite(Y)).any((1, 2)).sum()),
        "negative": int((Y < 0).any((1, 2)).sum()),
        "health rises": int((np.diff(H, axis=1) > 1e-12).any((1, 2)).sum()),
        "alive rises": int((np.diff(A, axis=1) > 1e-12).any((1, 2)).sum()),
        "zombies (alive > 0.01, health 0)": int(((A > 0.01) & (H <= 0)).any((1, 2)).sum()),
        "alive type absent at start appears": int(((A > 0) & (A[:, :1] == 0)).any((1, 2)).sum()),
        "mean health per unit above full by > 1%": int(
            ((H > 1.01 * A) & (A > 1e-3) & (H > 1e-3)).any((1, 2)).sum()
        ),
        "distance jumps > 2 per 0.125 s": int((np.abs(np.diff(d, axis=1)) > 2).any(1).sum()),
        "distance > 40": int((d > 40).any(1).sum()),
    }


head("1 fuzz: 5,000 random states, 8 s; ordinary, then extreme")
for name, m in M.items():
    with np.errstate(all="ignore"):
        for lab, kw in (
            ("ordinary", {}),
            ("huge armies", dict(lo=5, hi=50)),
            ("one unit", dict(lo=0.1, hi=0.1, zero=0.5)),
            ("distance 0-0.1", dict(dmin=0, dmax=0.1)),
            ("distance 100-1000", dict(dmin=100, dmax=1000)),
            ("tiny fractions", dict(lo=1e-12, hi=1e-6)),
        ):
            B = batch(m, 5000 if lab == "ordinary" else 1000, 64, **kw)
            r = check(m, B, m.rollout(B))
            print(f"{name:9s} {lab:18s}", {k: v for k, v in r.items() if v} or "clean")

head("2 long rollouts (90 s): does anything break late?")
for name, m in M.items():
    B = batch(m, 1000, 720, zero=0.0, lo=0.3)
    with np.errstate(all="ignore"):
        Y = m.rollout(B)
    T = m.L.team_view(Y)
    over = ((T[..., 0] < 0.01) | (T[..., 1] < 0.01)).any(1)
    wd = B["U"][:, 0, 2] == 1
    print(
        name,
        {k: v for k, v in check(m, B, Y).items() if v} or "clean",
        f"| battle never decided in 90 s: not withdrawing {(~over[~wd]).mean():.1%}, withdrawing {(~over[wd]).mean():.1%}",
    )

head("3 purity: input unchanged, batch = one by one, repeat calls equal, state carried between calls")
for name, m in M.items():
    B = batch(m, 40, 32)
    B0 = copy.deepcopy(B)
    Y = m.rollout(B)
    one = np.concatenate([m.rollout({k: v[i : i + 1] for k, v in B.items()}) for i in range(40)])
    m.rollout(batch(m, 7, 8))
    print(
        name,
        "input unchanged",
        all(np.array_equal(B[k], B0[k]) for k in B),
        "| batch vs single max diff",
        np.abs(Y - one).max(),
        "| repeat equal",
        np.array_equal(Y, m.rollout(B)),
    )

head("4 time splitting: 2 s + 2 s from the predicted state against 4 s in one go (reach offset restarts)")
for name, m in M.items():
    B = batch(m, 500, 32, zero=0.0, lo=0.3, dmin=4, dmax=10)
    Y = m.rollout(B)
    B2 = B | {"X0": Y[:, 16], "U": B["U"][:, 16:]}
    sh = m._shares(B2["X0"], B["shape0"], B["front0"])
    B2["reach0"] = np.column_stack(
        [B2["X0"][:, m.L.idx("alive", s, t)] * sh[(s, t)] for s in SIDES for t in m.L.types]
    )
    d = np.abs(m.L.team_view(m.rollout(B2))[:, -1, :2] - m.L.team_view(Y)[:, -1, :2])
    print(name, "health difference after 4 s: mean", d.mean().round(4), "max", d.max().round(4))

head("5 NaN / inf in one element: does it stay in its own row?")
for name in ("marines", "triple"):
    m = M[name]
    for key in ("X0", "shape0", "front0", "reach0", "U"):
        for bad in (np.nan, np.inf):
            B = batch(m, 20, 16, zero=0.0, lo=0.3)
            ref = m.rollout(B)
            B[key] = B[key].copy()
            B[key][(3,) + (0,) * (B[key].ndim - 1)] = bad
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    Y = m.rollout(B)
                    oth = np.delete(np.arange(20), 3)
                    print(
                        f"{name} {key} {bad}: other rows unchanged {np.array_equal(Y[oth], ref[oth])}; bad row finite {np.isfinite(Y[3]).all()}"
                    )
                except Exception as e:
                    print(name, key, bad, "RAISES", type(e).__name__, str(e)[:60])

head("6 geometry: states under translation, rotation, reflection, relabelling, float32")


def army(n, ty, seed=0, types=("MARINE", "MARAUDER", "STALKER")):
    r = np.random.default_rng(seed)
    return pd.DataFrame(
        [
            {
                "step": 0,
                "team": s,
                "unit_id": i,
                "unit_type": types[i % ty],
                "x": x + r.normal(0, 1.2),
                "y": 10 + r.normal(0, 1.5),
                "hp": 20.0 + i,
                "shield": 0.0,
                "alive": True,
                "target_team": None,
                "target_id": -1,
                "target_in_range": False,
            }
            for s, x in (("blue", 8.0), ("red", 14.0))
            for i in range(n)
        ]
    )


TY = ("MARINE", "MARAUDER", "STALKER")
base = typed_states(army(9, 3), TY, max_reach=6.0).iloc[0]


def rel(u):
    return float(
        np.nanmax(np.abs(typed_states(u, TY, max_reach=6.0).iloc[0].to_numpy(float) - base.to_numpy(float)))
    )


u = army(9, 3)
print("translate 1e3", rel(u.assign(x=u.x + 1000, y=u.y - 500)))
for a in (0.3, np.pi / 2, 2.0):
    print(
        "rotate",
        round(a, 2),
        rel(u.assign(x=np.cos(a) * u.x - np.sin(a) * u.y, y=np.sin(a) * u.x + np.cos(a) * u.y)),
    )
print(
    "reflect",
    rel(u.assign(y=-u.y)),
    "| shuffled rows",
    rel(u.sample(frac=1, random_state=0)),
    "| relabelled ids",
    rel(u.assign(unit_id=u.unit_id.map(lambda i: 8 - i))),
    "| type order in the call",
    "n/a",
)
print(
    "all units at one point:",
    typed_states(u.assign(x=1.0, y=1.0), TY, max_reach=6.0)
    .iloc[0][["distance", "depth_blue", "reach_blue_marine"]]
    .to_dict(),
)
try:
    typed_states(army(9, 3), TY[:2], max_reach=6.0)
    print("a third type present but not asked for: ACCEPTED (its positions count, its health does not)")
except ValueError as e:
    print("a third type present but not asked for: REFUSED:", e)

head("7 reach at the boundary: exactly 6, 5.75, just inside / outside")
for gap in (5.75, 5.75 + 1e-9, 5.75 - 1e-9, 6.0):
    uu = pd.DataFrame(
        [
            {
                "step": 0,
                "team": s,
                "unit_id": 0,
                "unit_type": "MARINE",
                "x": q,
                "y": 0.0,
                "hp": 45.0,
                "shield": 0.0,
                "alive": True,
                "target_team": None,
                "target_id": -1,
                "target_in_range": False,
            }
            for s, q in (("blue", 0.0), ("red", gap))
        ]
    )
    r = typed_states(uu, ("MARINE",), max_reach=6.0).iloc[0]
    print(gap, "in reach", r["reach_blue_marine"] * 10, "depth", r["depth_blue"])

head("8 sensitivity: how much does the 2 s prediction move for a small change in the state?")
for name, m in M.items():
    B = batch(m, 2000, 16, zero=0.0, lo=0.3, dmin=3, dmax=12)
    Y = m.L.team_view(m.rollout(B))
    out = {}
    for key, eps in (("X0", 0.01), ("shape0", 0.1), ("front0", 0.1), ("reach0", 0.01)):
        B2 = {k: v.copy() for k, v in B.items()}
        B2[key] += eps * rng.choice([-1, 1], B2[key].shape)
        if key == "X0":
            B2[key] = np.maximum(B2[key], 0)
        D = np.abs(m.L.team_view(m.rollout(B2))[:, -1, :2] - Y[:, -1, :2])
        out[f"{key} ±{eps}"] = f"mean {D.mean():.4f}, 99.9% {np.quantile(D, 0.999):.3f}"
    print(name, out)

head("9 intents that should and should not matter")
for name, m in M.items():
    B = batch(m, 2000, 24, zero=0.0, lo=0.3, dmin=3, dmax=10)
    Y = {}
    for it in (*INTENTS, "withdraw/focus"):
        Y[it] = m.L.team_view(m.rollout(B | {"U": np.repeat(plan_controls(it, 3.0)[None], 2000, 0)}))[:, -1]
    mg = {k: v[:, 0] - v[:, 1] for k, v in Y.items()}
    print(
        name,
        "focus ever predicted worse than nearest (advance):",
        f"{(mg['advance/focus'] < mg['advance/nearest'] - 1e-9).mean():.1%}",
        "| withdraw/focus = withdraw/nearest:",
        f"{(np.abs(mg['withdraw/focus'] - mg['withdraw/nearest']) < 1e-9).mean():.0%}",
        "| Red loses health while Blue withdraws:",
        f"{(B['X0'][:, :1] >= 0).mean() and (np.abs(Y['withdraw/nearest'][:, 1] - m.L.team_view(B['X0'])[:, 1]) > 1e-9).mean():.0%}",
    )

head("10 world-model interface: hostile input")
wm = WorldModel.load("marines")
U0 = [
    {"team": t, "unit_type": "MARINE", "x": x + 0.8 * (i % 3), "y": float(i), "hp": 45.0}
    for t, x in (("blue", 0.0), ("red", 8.0))
    for i in range(5)
]


def tryit(lab, f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            r = f()
            print(
                f"{lab:38s} -> ACCEPTED {str(r)[:110]}" + (f"  [warns {w[0].category.__name__}]" if w else "")
            )
        except Exception as e:
            print(f"{lab:38s} -> {type(e).__name__}: {str(e)[:100]}")


edit = lambda **kw: [U0[0] | kw, *U0[1:]]  # noqa: E731
hp = lambda us: (
    wm.predict(wm.state_from_units(us), "advance", 2.0).end[["hp_blue", "hp_red"]].round(1).tolist()
)  # noqa: E731
tryit("baseline", lambda: hp(U0))
tryit("hp above maximum (4500)", lambda: hp(edit(hp=4500.0)))
tryit("hp negative", lambda: hp(edit(hp=-5.0)))
tryit("hp inf", lambda: hp(edit(hp=np.inf)))
tryit("hp NaN", lambda: hp(edit(hp=np.nan)))
tryit("hp as text", lambda: hp(edit(hp="45")))
tryit("x inf", lambda: hp(edit(x=np.inf)))
tryit("x None", lambda: hp(edit(x=None)))
tryit("shield on a marine (100)", lambda: hp(edit(shield=100.0)))
tryit("shield negative", lambda: hp(edit(shield=-100.0)))
tryit("alive False with hp 45", lambda: hp(edit(alive=False)))
tryit("alive missing for the others (NaN)", lambda: hp([U0[0] | {"alive": True}, *U0[1:]]))
tryit("team 'Blue' (capital)", lambda: hp(edit(team="Blue")))
tryit("team NaN", lambda: hp(edit(team=np.nan)))
tryit("unit_type None", lambda: hp(edit(unit_type=None)))
tryit("unit_type 'Marine'", lambda: hp([u | {"unit_type": "Marine"} for u in U0]))
tryit("extra column", lambda: hp([u | {"step": 7, "unit_id": 3, "front": 1} for u in U0]))
tryit("same unit listed twice", lambda: hp([U0[0], *U0]))
tryit("empty list", lambda: hp([]))
tryit("None", lambda: hp(None))
tryit("1 vs 1", lambda: hp([U0[0], U0[5]]))
tryit("1 vs 200", lambda: hp([U0[0], *[U0[5] | {"y": float(i)} for i in range(200)]]))
tryit("all at the same point", lambda: hp([u | {"x": 0.0, "y": 0.0} for u in U0]))
tryit("10,000 apart", lambda: hp([u | {"x": u["x"] + (1e4 if u["team"] == "red" else 0)} for u in U0]))
s = wm.state_from_units(U0)
tryit("horizon 0", lambda: wm.predict(s, "hold", 0))
tryit("horizon negative", lambda: wm.predict(s, "hold", -1))
tryit("horizon NaN", lambda: wm.predict(s, "hold", np.nan))
tryit("horizon inf", lambda: wm.predict(s, "hold", np.inf))
tryit("horizon 0.2 (not a multiple)", lambda: wm.predict(s, "hold", 0.2).horizon)
tryit("horizon 1000 s", lambda: len(wm.predict(s, "hold", 1000).trajectory))
tryit("horizon as text", lambda: wm.predict(s, "hold", "2"))
tryit("plan None", lambda: wm.predict(s, None, 2))
tryit("plan 5", lambda: wm.predict(s, 5, 2))
tryit("plan ['hold', 'advance'] (2 intents?)", lambda: wm.predict(s, ["hold", "advance"], 2).plan)
tryit("plan [('hold', -1)]", lambda: wm.predict(s, [("hold", -1.0)]))
tryit("plan [('hold', 'focus')] (typo)", lambda: wm.predict(s, [("hold", "focus")]))
tryit("plan ('hold', 2.0)", lambda: wm.predict(s, ("hold", 2.0)))
tryit("compare with no plans", lambda: len(wm.compare(s, [])))
tryit("compare commit > horizon", lambda: wm.compare(s, commit=99).shape)
tryit("compare commit 0", lambda: wm.compare(s, commit=0).shape)
tryit("compare horizon 2.05, commit 2", lambda: wm.compare(s, horizon=2.05).shape)
tryit(
    "predict mutates the state?", lambda: (s.row.copy().equals(s.row), wm.predict(s, "hold", 1) is not None)
)
tryit("state of another model", lambda: WorldModel.load("mixed").predict(s, "hold", 1))
tryit("a string instead of a state", lambda: wm.predict("state", "hold", 1))
print("DONE")
