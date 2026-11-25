# ruff: noqa: E402, E501, E731, B018, B023, B905, I001
# %% [markdown]
# # RQ2. State representation
#
# > *How does the choice of state representation affect the accuracy, complexity, and interpretability of the learned world model?*
#
# The project description asks to compare at least two of: Level A (H_B, H_R, d), Level B (groups), Level C (per unit). Until now the
# comparisons were spread over different data and metrics; section 2 puts A, A+ and B on the **same start points and metric**.
#
# | section | question | data | status |
# |---|---|---|---|
# | 1 | which quantity drives the losses? | marines, train, one step | exploratory |
# | 2 | Level A, A+ and B on the same battles | mixed and triple, fit train, score val | **new, exploratory** |
# | 3 | Level C: per unit | recorded (`generalisation/14`, `21`) | pre-registered rules, stopped |
# | 4 | added states that did not help | recorded | mixed |
#
# The answer is left for you (last cell).

# %%
import sys

sys.path.insert(0, "../lib")
import json
import rq
from rq import BLUE, GREY, H, INK, ORANGE, np, pd, plt
from IPython.display import display

from evaluate import team_rates
from sindy_wm.evaluation.explore_models import AliveVariantA
from sindy_wm.evaluation.harness_models import Lanchester
from sindy_wm.models.intent_model import DT, stlsq

M = rq.models()

# %% [markdown]
# ## 1. What drives the losses?
# One-variable fits through the origin of each side's health rate (marines, train) on the opponent's health, units alive, and units in reach
# (measured). Blue's withdrawing steps are left out of Red's losses, as Blue does not fire then.

# %%
tr = rq.rates("marines")
rows = []
for s, o in (("blue", "red"), ("red", "blue")):
    d = tr if s == "blue" else tr[tr.withdraw == 0]
    y = d[f"d_health_{s}_marine"].to_numpy()
    for v in ("health", "alive", "reach"):
        x = d[f"{v}_{o}_marine"].to_numpy()
        k = (x @ y) / (x @ x)
        rows.append(
            {
                "losses of": s,
                "opponent's": v,
                "R²": 1 - ((y - k * x) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                "k": -k,
            }
        )
drv = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(6.5, 2.8))
for j, (v, c) in enumerate((("health", GREY), ("alive", ORANGE), ("reach", BLUE))):
    q = drv[drv["opponent's"] == v].set_index("losses of")["R²"]
    b = ax.bar(np.arange(2) + (j - 1) * 0.26, q[["blue", "red"]], 0.24, color=c, label=f"opponent's {v}")
    ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
ax.margins(y=0.15)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks([0, 1], ["Blue's losses", "Red's losses"])
ax.set_ylabel("one-step R²")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
drv.pivot(index="losses of", columns="opponent's", values=["R²", "k"]).round(3)

# %% [markdown]
# ## 2. Level A, A+ and B on the same battles
# | level | state | model |
# |---|---|---|
# | A | H_B, H_R, d only | degree-2 library (10 terms) × {1, hold, wd, focus}, STLSQ at 0.15 (as the pipeline) and 0.02 |
# | A+ | team health, alive, distance; reach from a closure with formation depth and width | team A\*-2k |
# | B | the same, per unit type | pipeline |
#
# A and the baselines get the measured team state at the start; A+ and B also get measured reach and formation (held fixed). Health in
# ten marines' hp, mean per side. New fit, train only; val was also used for earlier choices.


# %%
def phi(x):
    b, r, d = x.T
    return np.column_stack([np.ones_like(b), b, r, d, b * b, b * r, b * d, r * r, r * d, d * d])


PHI = ["1", "B", "R", "d", "B²", "BR", "Bd", "R²", "Rd", "d²"]


def library(x, u):
    p = phi(x)
    c = np.column_stack([np.ones(len(u)), u[:, 1], u[:, 2], u[:, 3]])
    return (p[:, :, None] * c[:, None, :]).reshape(len(x), -1)


NAMES = [f"{p}" if c == "1" else f"{p}·{c}" for p in PHI for c in ("1", "hold", "wd", "focus")]


class LevelA:
    """x = [H_B, H_R, d] with intents; degree-2 SINDYc."""

    def __init__(self, thr):
        self.thr = thr

    def fit(self, T):
        X = T[["health_blue", "health_red", "distance"]].to_numpy()
        U = T[["advance", "hold", "withdraw", "focus"]].to_numpy()
        Th = library(X, U)
        self.c = np.column_stack(
            [stlsq(Th, T[k].to_numpy(), self.thr) for k in ("d_health_blue", "d_health_red", "d_distance")]
        )
        return self

    def rollout(self, B):
        x = B["X0"][:, [0, 1, 4]].copy()
        out = np.repeat(B["X0"][:, None], B["U"].shape[1] + 1, 1).copy()
        for n in range(B["U"].shape[1]):
            x = np.clip(x + DT * library(x, B["U"][:, n]) @ self.c, 0, [1e3, 1e3, 1e3])
            out[:, n + 1, [0, 1, 4]] = x
        return out


tab, fitted, ex = [], {}, {}
for name in ("mixed", "triple"):
    m = M[name]
    T = team_rates(rq.rates(name), m.L)
    B = m.L.batch(rq.states(name))
    tb = B["team"]
    A015, A002 = LevelA(0.15).fit(T), LevelA(0.02).fit(T)
    team = AliveVariantA(law="reach").fit(H.FitData(states=None, rates=T))
    lan = Lanchester().fit(H.FitData(states=None, rates=T))
    rolls = {
        "persistence": np.repeat(tb["X0"][:, None], tb["U"].shape[1] + 1, 1),
        "linear extrapolation": rq.last_slope(rq.states(name), B, m.L),
        "Lanchester (H only)": lan.rollout(tb),
        "A, threshold 0.15": A015.rollout(tb),
        "A, threshold 0.02": A002.rollout(tb),
        "A+ team A*-2k": team.rollout(tb),
        "B per type": rq.rollout(m, B),
    }
    t = rq.horizon_table(rolls, tb)
    terms = {
        "Lanchester (H only)": 2,
        "A, threshold 0.15": int((A015.c != 0).sum()),
        "A, threshold 0.02": int((A002.c != 0).sum()),
        "A+ team A*-2k": len(team.coefficients()),
        "B per type": m.complexity()["terms"],
    }
    q = t[["health_1s", "health_2s", "health_4s", "margin_r2_2s", "margin_r2_4s", "distance_2s"]].copy()
    q.insert(0, "terms", pd.Series(terms))
    q.insert(0, "set", name)
    tab.append(q)
    fitted[name] = (A015, A002, team)
    ex[name] = (rolls, tb)
levels = pd.concat(tab).reset_index(names="model").set_index(["set", "model"])
levels.round(4)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
st = {
    "persistence": (GREY, ":"),
    "linear extrapolation": (GREY, "-."),
    "Lanchester (H only)": (GREY, "--"),
    "A, threshold 0.15": (INK, "--"),
    "A, threshold 0.02": (INK, "-"),
    "A+ team A*-2k": (ORANGE, "-"),
    "B per type": (BLUE, "-"),
}
for ax, name in zip(axes, ("mixed", "triple")):
    rolls, tb = ex[name]
    for k, Y in rolls.items():
        e = []
        for h in (0.5, 1, 2, 3, 4):
            n = round(h / DT)
            ok = ~np.isnan(tb["Y"][:, n, 0])
            e.append(np.abs(Y[ok, n, :2] - tb["Y"][ok, n, :2]).mean())
        ax.plot((0.5, 1, 2, 3, 4), e, color=st[k][0], ls=st[k][1], label=k)
    ax.set_title(f"{name} (val)")
    ax.set_xlabel("horizon (s)")
axes[0].set_ylabel("health error, mean per side")
axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# %% [markdown]
# The A, A+ and B curves are close. The same data relative to Level B (above 1 = worse than B):

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3), sharey=True)
for ax, name in zip(axes, ("mixed", "triple")):
    rolls, tb = ex[name]
    err_ = lambda Y, h: np.abs(
        (Y - tb["Y"])[~np.isnan(tb["Y"][:, round(h / DT), 0]), round(h / DT), :2]
    ).mean()
    for k, c, ls in (
        ("A, threshold 0.15", INK, "--"),
        ("A, threshold 0.02", INK, "-"),
        ("A+ team A*-2k", ORANGE, "-"),
    ):
        ax.plot(
            (0.5, 1, 2, 3, 4),
            [err_(rolls[k], h) / err_(rolls["B per type"], h) for h in (0.5, 1, 2, 3, 4)],
            color=c,
            ls=ls,
            marker="o",
            ms=4,
            label=k,
        )
    ax.axhline(1, color=BLUE, lw=1.2)
    ax.set_title(f"{name} (val)")
    ax.set_xlabel("horizon (s)")
axes[0].set_ylabel("health error ÷ Level B's")
axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# %% [markdown]
# **Paired differences at 2 s and 4 s** (health error, resampled by battle, 95%; below 0 = the first is better).


# %%
def by_battle(Y, tb, h):
    n = round(h / DT)
    ok = ~np.isnan(tb["Y"][:, n, 0])
    e = np.abs(Y[ok, n, :2] - tb["Y"][ok, n, :2]).mean(1)
    return pd.Series(e).groupby(tb["ep"][ok]).agg(["sum", "size"])


def ratio_ci(a, b, n=1000):
    rng = np.random.default_rng(0)
    N = len(a)
    d = []
    av, bv, sz = a["sum"].to_numpy(), b["sum"].to_numpy(), a["size"].to_numpy()
    for _ in range(n):
        i = rng.integers(0, N, N)
        d.append((av[i].sum() - bv[i].sum()) / sz[i].sum())
    return (av.sum() - bv.sum()) / sz.sum(), *np.quantile(d, [0.025, 0.975])


out = []
for name in ("mixed", "triple"):
    rolls, tb = ex[name]
    for a, b in (
        ("A+ team A*-2k", "A, threshold 0.02"),
        ("B per type", "A+ team A*-2k"),
        ("B per type", "A, threshold 0.02"),
    ):
        for h in (2, 4):
            d, lo, hi = ratio_ci(by_battle(rolls[a], tb, h), by_battle(rolls[b], tb, h))
            out.append(
                {
                    "set": name,
                    "comparison": f"{a} − {b}",
                    "horizon": h,
                    "difference": d,
                    "low": lo,
                    "high": hi,
                }
            )
pd.DataFrame(out).round(4)

# %% [markdown]
# **The strict Level A equations** (mixed, threshold 0.15), as fitted:

# %%
A = fitted["mixed"][0]
for j, lab in enumerate(("H_B'", "H_R'", "d'")):
    print(lab, "=", "  ".join(f"{c:+.3f} {n}" for n, c in zip(NAMES, A.c[:, j]) if c))
print(
    "\nthreshold 0.02:",
    {lab: int((fitted["mixed"][1].c[:, j] != 0).sum()) for j, lab in enumerate(("H_B'", "H_R'", "d'"))},
    "terms",
)

# %% [markdown]
# ## 3. Level C: each unit on its own (recorded)
# A sparse law per unit failed its first stage (val R² 0.13-0.57 with 11-62 terms, `generalisation/14`). The hybrid model (walk / stop switch per
# unit, frozen damage) is compared with the frozen pipeline on the same val start points: **ratio of errors, below 1 = better than the pipeline**.

# %%
d = pd.DataFrame(json.load(open(rq.PROC / "generalisation" / "overnight20.json"))["h12"])
piv = (
    d[d.variant.isin(["base", "R", "SR"])]
    .pivot_table(index=["scenario", "quantity", "variant"], columns="h", values="ratio")
    .round(3)
)
display(piv)
print(
    "base = with no-overlap step, R = SMAClite's steering, S = switch learned from sequences; the rule asked for a reach ratio <= 0.7"
)

# %% [markdown]
# ## 4. Added states that did not improve the model (recorded)
# | added state | where | result |
# |---|---|---|
# | health spread (evenness) | `planning/10` | tiny val gain; same outcome in 95-100% of battles |
# | healthy / wounded / dead | `planning/14`, `16` | bolted on: no better; discovered: better only mixed 1 s |
# | reach as a state, generic library | `generalisation/08` | 54-232 terms; not sparse |
# | mean and spread of reach margins | `generalisation/11` | 26-51 groups; better at 1-2 s but 12-18% blow-ups |
# | share in reach (S, marines) | `final_exploration` | best margin R² at 4 s (0.41 vs 0.01 cv), 22 terms, 2 wrong signs |
#
# For scale: a black box on the per-type state (plus margins) reaches margin R² at 4 s of 0.61 (marines) and 0.54 (mixed) on val, against 0.08 and 0.19
# for the pipeline (`planning/37`).

# %% [markdown]
# ## Caveats
# - A+ and B are given measured reach and formation at the start; A is not. So they get more information, yet A is not worse at 4 s (tables above).
# - A\*-2k and the pipeline contain closures fitted outside SINDy; their terms are not their only numbers (RQ1, section 2).
# - Strict A is not sparse at either threshold (39-53 and 107-109 of 120 candidate terms), and its terms have no physical reading.
# - Val was used for earlier choices; nothing in section 2 was pre-registered.
# - With one unit type, A+ and B are nearly the same model (marines: identical health terms).
# - Two findings seem to pull apart: flexible predictors on the team state do much better (the state is not the limit), while true team-level shape removes only
#   2-5% of the margin error (`planning/24`).

# %% [markdown]
# ## Your answer (to write)
# 1. What does each step A → A+ → B → C buy, in accuracy, and what does it cost in numbers and readability?
# 2. Is reach part of the *state*, or a way of computing the right variable from the state?
# 3. Is the limit the state or the closure? Which number above supports you, and which speaks against you?
#
# *Answer:*
