# ruff: noqa: E402, E501, E731, B018, B023, B905, I001
# %% [markdown]
# # RQ4. Applicability across scenarios
#
# > *Can a world model learned from one set of combat scenarios also be used in other scenarios with different force sizes, unit types, and unit properties?*
#
# Two different claims must be kept apart:
# - **the same model** used elsewhere without refitting (what RQ4 literally asks), and
# - **the same method** refitted in each scenario.
#
# | section | claim | data | status |
# |---|---|---|---|
# | 1 | same method: structure and numbers per scenario | held-out refits (recorded) | pre-registered |
# | 2 | same model, other unit mixes: every model on every val set it can read | val | exploratory, computed here |
# | 3 | same model, uneven mixes never seen | held-out (recorded) | pre-registered |
# | 4 | same model, other force sizes | marines (recorded) | exploratory |
# | 5 | same model, another unit type by scaling | stalkers test2 (recorded) | pre-registered (candidates A\*, B, S) |
# | 6 | what breaks when types are added | train | exploratory |
#
# The answer is left for you (last cell).

# %%
import sys

sys.path.insert(0, "../lib")
import rq
from rq import BLUE, INK, np, pd, plt
from triple import with_missing_types

M, T = rq.models(), rq.held_out()

# %% [markdown]
# ## 1. Same method, refitted
# Expectations met in the pre-registered test (E1 structure, E2 physics, E3 beats persistence and Lanchester, E4 beats team level, E5 horizon, E6 val ≈ test).

# %%
exp = pd.DataFrame({s: {k.split()[0]: v for k, v in T[s]["expectations"].items()} for s in T}).T
print(
    "met:",
    int(exp.apply(lambda c: c.map(lambda v: v is True)).sum().sum()),
    "of",
    int(exp.notna().sum().sum()),
)
exp.replace({True: "met", False: "MISSED"}).fillna("–")

# %% [markdown]
# The same firepower number in different scenarios (train fits; attacker → target, Blue's losses / Red's losses):

# %%
rows = {}
for name, m in M.items():
    for a in m.L.types:
        for t in m.L.types:
            rows.setdefault(f"{a} → {t}", {})[name] = " / ".join(
                f"{-m.coef[(s, t)][f'R[{a}]*s[{t}]']:.3f}" for s in ("blue", "red")
            )
pd.DataFrame(rows).T.reindex(columns=list(M)).fillna("")

# %% [markdown]
# ## 2. Same model, other unit mixes (val, no refitting)
# Rows: the battles. Columns: the model used, unchanged. A model can only read battles whose unit types it knows. Health error at 2 s **relative to the
# model fitted on those battles** (1.00 = as good as its own model), and the margin R² at 2 s.


# %%
def score(m, sv, scale):
    B = m.L.batch(sv)
    t = rq.horizon_table({"x": rq.rollout(m, B)}, B["team"], scale)
    return t.loc["x"]


rel, r2, err = {}, {}, {}
for s in M:
    sv = rq.states(s)
    ref_hp = M[s].L.hp[M[s].L.types[0]]
    own = score(M[s], sv, ref_hp)
    for name, m in M.items():
        if set(M[s].L.types) <= set(m.L.types):
            q = score(m, with_missing_types(sv, m.L.types), m.L.hp[m.L.types[0]])
            rel.setdefault(s, {})[name] = q.health_2s / own.health_2s
            r2.setdefault(s, {})[name] = q.margin_r2_2s
            err.setdefault(s, {})[name] = q.health_4s / own.health_4s
R2, REL, REL4 = (pd.DataFrame(x).T.reindex(index=list(M), columns=list(M)) for x in (r2, rel, err))
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
for ax, D, title, lo, hi in (
    (axes[0], REL, "health error at 2 s ÷ own model's", 1.0, 1.4),
    (axes[1], R2, "margin R² at 2 s", 0.0, 0.5),
):
    ax.imshow(
        np.ma.masked_invalid(D.to_numpy(float)), cmap="Blues", vmin=lo - 0.1, vmax=hi + 0.25, aspect="auto"
    )
    ax.grid(False)
    for i in range(5):
        for j in range(5):
            v = D.iloc[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=INK,
                    fontweight="bold" if i == j else "normal",
                )
    ax.set_xticks(range(5), list(M), fontsize=8)
    ax.set_yticks(range(5), list(M), fontsize=8)
    ax.set_xlabel("model used")
    ax.set_title(title)
axes[0].set_ylabel("battles (val)")
plt.tight_layout()
plt.show()
print("health error at 4 s ÷ own model's")
REL4.round(2)

# %% [markdown]
# Each model keeps its own withdraw radius (7.0 marines, 8.67 marauders, 8.83 stalkers, 8.0 mixed and triple), so part of any difference is that setting.

# %% [markdown]
# ## 3. Same model, uneven mixes (held-out, recorded)
# The mixed model, unchanged, on battles where one side has 2/3 marauders and the other 1/4: never seen in fitting.

# %%
pd.DataFrame(T["uneven"]["table"])[
    ["health_1s", "health_2s", "health_4s", "margin_r2_2s", "margin_r2_4s"]
].round(4)

# %% [markdown]
# ## 4. Same model, other force sizes (marines, recorded)
# Fitted on battles of at most 25 units, scored on 26-33. Compared with the ordinary cross-validation of the same models.

# %%
c = pd.read_csv(rq.DOCS / "final_exploration" / "results.csv")
g = (
    c[c.split.isin(["cv", "size"])]
    .groupby(["experiment", "split"])[["health_4s", "margin_r2_4s", "battle_health_nrmse"]]
    .mean()
    .unstack("split")
)
g.loc[["ref/persistence", "ref/lanchester-square", "ref/A", "deep/A*", "ref/B", "ref/S", "J/mlp-wide"]].round(
    3
)

# %% [markdown]
# Larger battles have more health on the same scale (ten marines = 1), so errors grow with size for every model; compare with persistence.

# %% [markdown]
# ## 5. Another unit type by scaling (stalkers, test2, recorded)
# The marine models scaled by the game's hp, reach, cooldown and speed, no refitting. Health in ten stalkers.

# %%
s = c[c.split == "test2_stalkers"].groupby("experiment")[["health_2s", "health_4s", "margin_r2_4s"]].mean()
s.round(3)

# %% [markdown]
# ## 6. What breaks when types are added
# Effect size of every firepower term (train, all other real terms included). The threshold is 0.15.

# %%
import threshold as TH

fig, ax = plt.subplots(figsize=(8, 3))
for j, name in enumerate(TH.NAMES):
    m, nb, F = TH.grams(name)
    e = []
    for k, (names, G) in F.items():
        A, b, sd, n = TH.scaled(G.sum(0), names)
        i = [names.index(f"R[{a}]*s[{k[1]}]") for a in m.L.types]
        e += list(np.abs(np.linalg.solve(A[np.ix_(i, i)], b[i])) / sd[len(names)])
    ax.scatter(e, j + np.linspace(-0.2, 0.2, len(e)), color=BLUE, s=34, edgecolor="white", lw=0.8)
ax.axvline(0.15, color=INK, lw=0.8, ls="--")
ax.text(0.155, -0.4, "0.15", fontsize=8)
ax.set_yticks(range(5), TH.NAMES)
ax.invert_yaxis()
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_xlabel("effect size of the term")
plt.tight_layout()
plt.show()

# %% [markdown]
# With three types the exact structure appears in 41 of 60 bootstrap resamples, against 60 of 60 for every other model (RQ1, section 4).

# %% [markdown]
# ## Caveats
# - Unit properties were never varied one at a time: types differ in hp, armor, damage, cooldown, range, size and speed together.
# - One open map, the scripted Red AI, 6-14 Red units, close battles (strength ratio 0.85-1.18).
# - Every type is a ranged ground unit; melee, splash, healers and structures are untested.
# - Section 2 is val only and was not pre-registered. The triple model was fitted on 1500 battles, the mixed and marine models on 6000.
# - Stalkers alone: advance speed 0.48 × the game, and shields regenerate, which the model cannot express.
# - Only S4-like underdog battles show what happens outside the trained strength ratio, and only in planning (RQ5).

# %% [markdown]
# ## Your answer (to write)
# 1. Is your claim that the *model* transfers or that the *method* does? Which section carries each?
# 2. For which changes (mix, size, type, property) do you have evidence, and for which none?
# 3. What would a reader need to see to believe it works on VR-Forces?
#
# *Answer:*
