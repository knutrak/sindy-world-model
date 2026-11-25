# ruff: noqa: E402, E501, E731, B018, B023, B905, I001
# %% [markdown]
# # RQ3. Predictive performance
#
# > *How accurately can the identified world model predict future game states over different planning horizons?*
#
# | section | question | data | status |
# |---|---|---|---|
# | 1 | error against horizon and baselines | four held-out sets (recorded) | pre-registered, run once |
# | 2 | how far is it reliable? | the same | pre-registered rule |
# | 3 | linear extrapolation, the missing baseline | val | new, exploratory |
# | 4 | which battles go wrong? | val | exploratory |
# | 5 | what a prediction looks like | val, chosen by rule | illustration |
# | 6 | the price of interpretability | marine cross-validation (recorded) | exploratory |
#
# The answer is left for you (last cell).

# %%
import sys

sys.path.insert(0, "../lib")
import rq
from rq import AQUA, BLUE, GREY, INK, ORANGE, np, pd, plt
from sindy_wm.models.intent_model import DT

M, T = rq.models(), rq.held_out()
SETS = ["marines", "marauders", "mixed", "uneven"]

# %% [markdown]
# ## 1. Error against horizon, held-out
# Health error = mean absolute error per side, in ten units of the scenario's first type, over start points in battles never used for fitting.

# %%
hs = [1, 2, 4, 8]
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
st = {
    "pipeline": (BLUE, "-"),
    "persistence": (GREY, ":"),
    "Lanchester": (GREY, "--"),
    "team-level A*-2k": (ORANGE, "-"),
}
for ax, s in zip(axes, SETS):
    t = pd.DataFrame(T[s]["table"])
    for k, (c, ls) in st.items():
        if k in t.index:
            ax.plot(hs, [t.loc[k, f"health_{h}s"] for h in hs], color=c, ls=ls, marker="o", ms=4, label=k)
    ax.set_title(f"{s} (held-out)")
    ax.set_xlabel("horizon (s)")
    ax.set_ylim(0)
axes[0].set_ylabel("health error")
axes[2].legend()
plt.tight_layout()
plt.show()
rows = []
for s in SETS:
    for k, v in T[s]["ci_model_minus_baseline"].items():
        rows.append({"set": s, "pipeline minus": k, "low": v[0], "high": v[1]})
pd.DataFrame(rows).pivot(index="set", columns="pipeline minus", values=["low", "high"]).round(4).T

# %% [markdown]
# ## 2. How far is it reliable?
# Margin R² = share of the variance of the *change* in (Blue − Red) health explained. The pre-registered rule: the longest horizon with R² ≥ 0.3 and the lower
# 95% bound above 0.

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for s, c, mk in zip(SETS, (BLUE, ORANGE, AQUA, "#eda100"), "osD^"):
    d = T[s]["horizon_detail"]
    x = [float(k.split()[0]) for k in d]
    y = [v["margin R2"] for v in d.values()]
    ax.plot(x, y, color=c, marker=mk, ms=5, label=f"{s} ({T[s]['horizon']} s)")
    ax.fill_between(
        x,
        [v["interval"][0] for v in d.values()],
        [v["interval"][1] for v in d.values()],
        color=c,
        alpha=0.12,
        lw=0,
    )
ax.axhline(0.3, color=INK, lw=0.8, ls="--")
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylim(-0.6, 0.6)
ax.set_xlabel("horizon (s)")
ax.set_ylabel("margin R²")
ax.legend(title="set (reliable horizon)")
plt.tight_layout()
plt.show()
pd.DataFrame({s: {k: round(v["margin R2"], 2) for k, v in T[s]["horizon_detail"].items()} for s in SETS}).T

# %% [markdown]
# Triple (exp12 held-out, run once on 25 Nov, `docs/triple/02_test.md`): health error 0.0426 at 2 s, 0.0778 at 4 s; margin R² 0.45 and 0.35; reliable
# horizon 4 s.

# %% [markdown]
# ## 3. Linear extrapolation (val)
# The project description names persistence *or* linear extrapolation as baselines; the held-out test had only persistence and Lanchester. Here the last
# second's rate simply continues.

# %%
rows, rolls = [], {}
for s in ("marines", "marauders", "stalkers", "mixed", "triple"):
    m, sv = M[s], rq.states(s)
    B = m.L.batch(sv)
    tb = B["team"]
    r = {
        "pipeline": rq.rollout(m, B),
        "linear extrapolation": rq.last_slope(sv, B, m.L),
        "persistence": np.repeat(tb["X0"][:, None], tb["U"].shape[1] + 1, 1),
    }
    rolls[s] = (B, r)
    t = rq.horizon_table(r, tb)
    for k in r:
        rows.append(
            {
                "set": s,
                "model": k,
                **{f"health {h} s": t.loc[k, f"health_{h}s"] for h in (1, 2, 4)},
                "distance 2 s": t.loc[k, "distance_2s"],
                "margin R² 2 s": t.loc[k, "margin_r2_2s"],
            }
        )
lin = pd.DataFrame(rows).set_index(["set", "model"])
lin.round(4)

# %% [markdown]
# ## 4. Which predictions go wrong? (val, 2 s)
# Absolute error of the predicted *margin change* (less changes while withdrawing or with few units, so those errors are smaller), by whether Blue withdraws during the 2 s, by how strung out Red is at the start, and by units left.

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
out = []
for s, c, mk in (("marines", BLUE, "o"), ("marauders", ORANGE, "s"), ("mixed", AQUA, "D")):
    B, r = rolls[s]
    tb = B["team"]
    n = round(2 / DT)
    ok = ~np.isnan(tb["Y"][:, n, 0])
    Y = r["pipeline"]
    g = lambda Z: Z[ok, n, 0] - Z[ok, n, 1] - (Z[ok, 0, 0] - Z[ok, 0, 1])
    e = np.abs(g(Y) - g(tb["Y"]))
    wd = tb["U"][ok, :n, 2].mean(1)
    dep = tb["shape0"][ok, 1]
    nu = 10 * tb["X0"][ok][:, 2:4].sum(1)
    for ax, v, q, lab in (
        (axes[0], wd, [-0.01, 0.001, 0.5, 1.01], "share of the 2 s withdrawing"),
        (axes[1], dep, np.quantile(dep, [0, 0.25, 0.5, 0.75, 1]), "Red depth (quartile)"),
        (axes[2], nu, [0, 4, 8, 14, 60], "units left, both sides"),
    ):
        b = pd.cut(v, np.unique(q), include_lowest=True)
        w = pd.Series(e).groupby(b, observed=True).mean()
        ax.plot(range(len(w)), w, color=c, marker=mk, ms=6, label=s)
        ax.set_xlabel(lab)
        ax.set_xticks(
            range(len(w)),
            ["none", "some", "most"][: len(w)]
            if ax is axes[0]
            else [f"Q{i + 1}" for i in range(len(w))]
            if ax is axes[1]
            else ["≤4", "5-8", "9-14", ">14"][: len(w)],
        )
for ax in axes:
    ax.set_ylim(0)
axes[0].set_ylabel("margin error at 2 s\n(own health unit)")
axes[0].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. What a prediction looks like
# Marines, val. Among start points with at least 4 s left, at least 4 units per side, and at least 10% of both sides' total health lost in the next 2 s
# (so there is a fight), sorted by health error at 2 s: the 10th, 50th and 90th percentile. Chosen by this rule, not by hand.

# %%
B, r = rolls["marines"]
tb = B["team"]
n2, n4 = round(2 / DT), round(4 / DT)
tot = tb["Y"][:, :, :2].sum(2)
ok = np.flatnonzero(
    ~np.isnan(tb["Y"][:, n4, 0])
    & (tb["X0"][:, 2:4].min(1) >= 0.4)
    & (tot[:, 0] - tot[:, n2] >= 0.1 * tot[:, 0])
)
print(len(ok), "start points qualify")
e2 = np.abs(r["pipeline"][ok, n2, :2] - tb["Y"][ok, n2, :2]).mean(1)
pick = ok[np.argsort(e2)[(np.array([0.1, 0.5, 0.9]) * (len(ok) - 1)).astype(int)]]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3), sharey=True)
t = np.arange(n4 + 1) * DT
for ax, i, lab in zip(axes, pick, ("10th percentile", "median", "90th percentile")):
    for j, c, side in ((0, BLUE, "Blue"), (1, ORANGE, "Red")):
        ax.plot(t, tb["Y"][i, : n4 + 1, j], color=c, label=f"{side}, game")
        ax.plot(t, r["pipeline"][i, : n4 + 1, j], color=c, ls="--", label=f"{side}, model")
    ax.axvline(2, color=INK, lw=0.6, ls=":")
    ax.set_title(f"{lab}: error {np.abs(r['pipeline'][i, n2, :2] - tb['Y'][i, n2, :2]).mean():.3f} at 2 s")
    ax.set_xlabel("s")
axes[0].set_ylabel("team health")
axes[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. The price of interpretability (marines, 5-fold cv, recorded)

# %%
c = pd.read_csv(rq.DOCS / "final_exploration" / "results.csv")
c = (
    c[c.split == "cv"]
    .groupby("experiment")[
        [
            "n_terms",
            "health_1s",
            "health_2s",
            "health_4s",
            "margin_r2_2s",
            "margin_r2_4s",
            "battle_health_nrmse",
        ]
    ]
    .mean()
)
c.loc[
    [
        "ref/persistence",
        "ref/lanchester-square",
        "ref/no-control",
        "ref/A",
        "deep/A*",
        "ref/B",
        "ref/S",
        "J/mlp-wide",
    ]
].round(3)

# %% [markdown]
# ## Caveats
# - Health error is a **mean over the two sides** here and in `harness.py`; `planning/37` reports the **sum** (twice as large).
# - Scales differ between sets (ten units of the first type), so compare within a set.
# - The formation is held at its starting value in every rollout; whole-battle durations are off by 3-6 s.
# - Winner accuracy is not used to rank models (a model without intents scored highest).
# - False stand-offs: 0.3-1.2% of 2-3 s predictions (`docs/world_model.md`).
# - Horizon 8 s is far beyond the reliable range; its margin R² is negative everywhere in section 2.

# %% [markdown]
# ## Your answer (to write)
# 1. State one accuracy number per horizon you would defend, with its interval and its set.
# 2. The use case needs 1-2 s. Does section 2 support that for every unit type?
# 3. Where does the model fail, and is that a failure of the equations, the closures, or the fixed formation?
#
# *Answer:*
