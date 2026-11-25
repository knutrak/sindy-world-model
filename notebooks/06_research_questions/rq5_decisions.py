# ruff: noqa: E402, E501, E731, B018, B023, B905, I001
# %% [markdown]
# # RQ5. Decision making with a world model (key question)
#
# > *Does providing an LLM-driven agent with access to a learned world model improve decision making compared to using the same LLM without a world model?*
#
# **No LLM has been used, so RQ5 itself is not answered.** It belongs to Phase 2. What exists is Phase 2's first step: a planner without an LLM, and a measurement of
# how much a perfect model would gain.
#
# | section | question | data | status |
# |---|---|---|---|
# | 1 | is there anything to gain? | Phase A, 200 paired seeds | pre-registered |
# | 2 | does the frozen model capture it? | Phase B, 400 paired seeds | pre-registered |
# | 3 | is it replicated? | two fresh seed sets | recorded |
# | 4 | the strongest simple rule and the matched ceiling | 400 paired seeds | pre-registered |
# | 5 | does better prediction give better decisions? | val and closed loop | pre-registered |
# | 6 | where do the models choose differently? | 8,594 re-plans | exploratory |
#
# S1 marines, S2 mixed, S3 Blue marauders against Red marines, S4 underdog, S5 uneven. **S3 and S4 lie outside the model's tested range** (flagged).
# The answer is left for you (last cell).

# %%
import sys

sys.path.insert(0, "../lib")
import json
import rq
from rq import AQUA, BLUE, GREY, INK, ORANGE, np, pd, plt

P = rq.PROC / "planning"
S = ["S1", "S2", "S3", "S4", "S5"]
BEST = {
    "S1": "fixed:hold/focus",
    "S2": "fixed:advance/focus",
    "S3": "fixed:hold/focus",
    "S4": "fixed:hold/focus",
    "S5": "fixed:advance/focus",
}


def load(kind):
    return {
        s: pd.read_csv(P / f"{kind}_{s}.csv")
        .pivot_table(index="seed", columns="policy", values="won", aggfunc="first")
        .astype(float)
        for s in S
    }


def compare(W, a, b):
    out = []
    for s in S:
        pa, pb = a(s), b(s)
        if pa not in W[s] or pb not in W[s]:
            continue
        d = W[s][[pa, pb]].dropna() if pa != pb else W[s][[pa]].dropna()
        m, lo, hi = rq.paired_ci((d[pa] - d[pb]).to_numpy())
        out.append(
            {
                "scenario": s,
                "policy": pa,
                "against": pb,
                "n": len(d),
                "wins": d[pa].mean(),
                "its wins": d[pb].mean(),
                "difference": m,
                "low": lo,
                "high": hi,
                "read-out": "beats" if lo > 0 else "loses" if hi < 0 else "ties",
            }
        )
    return pd.DataFrame(out).set_index("scenario")


# %% [markdown]
# ## 1. Is there anything to gain? (Phase A)
# A planner that uses *copies of the real game* as its model (exact oracle), and one whose copies have perturbed positions and health.

# %%
A = load("headroom")
display_rows = []
for pol in ("oracle", "oracle-perturbed"):
    display_rows.append(compare(A, lambda s, pol=pol: pol, lambda s: BEST[s]))
headroom = pd.concat(display_rows)
headroom.round(3)

# %% [markdown]
# ## 2. Does the frozen model capture it? (Phase B)
# Planner with the frozen pipeline model, horizon 3 s, against Phase A's best fixed intent (chosen before).

# %%
B = load("phase_b")
pb = compare(B, lambda s: "model:h3", lambda s: BEST[s])
pb.round(3)

# %%
fig, ax = plt.subplots(figsize=(7, 3))
for i, s in enumerate(S):
    r = pb.loc[s]
    c = GREY if s in ("S3", "S4") else BLUE
    ax.plot([r.low * 100, r.high * 100], [i, i], color=c)
    ax.plot(r.difference * 100, i, "o", color=c, ms=8)
    ax.text(r.high * 100 + 0.7, i, f"{r['read-out']}", va="center", fontsize=8)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(5), [f"{s} (flagged)" if s in ("S3", "S4") else s for s in S])
ax.invert_yaxis()
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_xlabel("planner wins − best fixed intent's wins (points, 95%)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Replications (recorded, `docs/planning/21_rq5_report.md`)
# | | Phase B | replication 1 | replication 2 |
# |---|---|---|---|
# | S1 | −3.8 loses | −2.2 ties | −4.0 loses |
# | S2 | +2.2 beats | −1.0 ties | +0.2 ties |
# | S3 (flagged) | +16.8 beats | +13.5 beats | +15.0 beats |
# | S4 (flagged) | −1.2 ties | +0.2 ties | −1.5 ties |
# | S5 | 0.0 ties | −0.5 ties | +0.2 ties |

# %% [markdown]
# ## 4. The strongest simple rule, and a ceiling with the planners' own settings
# Kite (shoot, step back) was found later and is not in the frozen model's data. "Perfect model" = the exact game with the planners' settings (3 s horizon, five
# team-level candidates).

# %%
C = load("closed34")
kite = {s: max(("fixed:kite/nearest", "fixed:kite/focus"), key=lambda k: C[s][k].mean()) for s in S}
wins = pd.DataFrame(
    {
        s: {
            "best fixed team intent": C[s][BEST[s]].mean(),
            "best fixed kite": C[s][kite[s]].mean(),
            "frozen planner": C[s]["model:h3"].mean(),
            "μ, σ planner (5 intents)": C[s]["musigma5:h3"].mean(),
            "MLP planner": C[s]["mlp:h3"].mean(),
            "perfect model, 3 s": C[s]["oracle5-h3"].mean(),
        }
        for s in S
    }
).T
fig, ax = plt.subplots(figsize=(9, 3.6))
for k, c, mk in (
    ("best fixed team intent", GREY, "s"),
    ("best fixed kite", INK, "D"),
    ("frozen planner", BLUE, "o"),
    ("μ, σ planner (5 intents)", ORANGE, "^"),
    ("perfect model, 3 s", AQUA, "*"),
):
    off = {
        "best fixed team intent": -0.24,
        "best fixed kite": -0.12,
        "frozen planner": 0.0,
        "μ, σ planner (5 intents)": 0.12,
        "perfect model, 3 s": 0.24,
    }[k]
    ax.plot(np.arange(5) + off, wins[k] * 100, ls="", marker=mk, ms=11 if mk == "*" else 8, color=c, label=k)
ax.set_xticks(range(5), [f"{s}*" if s in ("S3", "S4") else s for s in S])
ax.set_ylabel("Blue wins (%)")
ax.set_ylim(0, 102)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
print("* flagged: outside the tested range; kite is chosen here as the better of two on these seeds")
(wins * 100).round(1)

# %% [markdown]
# Share of the matched ceiling captured: (planner − best fixed) ÷ (perfect model − best fixed), team-level candidates only.

# %%
sh = pd.DataFrame(
    {
        k: (wins[k] - wins["best fixed team intent"])
        / (wins["perfect model, 3 s"] - wins["best fixed team intent"])
        for k in ("frozen planner", "μ, σ planner (5 intents)")
    }
)
t1 = compare(C, lambda s: "musigma5:h3", lambda s: BEST[s])[["difference", "low", "high", "read-out"]]
t3 = compare(C, lambda s: "oracle5-h3", lambda s: BEST[s])[["difference", "low", "high", "read-out"]]
pd.concat({"μ, σ − best fixed": t1, "perfect − best fixed": t3, "share captured": sh}, axis=1).round(3)

# %% [markdown]
# ## 5. Does better prediction give better decisions?
# Offline accuracy on val (`planning/37`) next to closed-loop wins. The black box had the same inputs, plus reach margins and kite among its candidates in mixed.

# %%
bb = pd.DataFrame(json.load(open(P / "black_box.json"))["val"])
off = bb[bb.h == 4].pivot(index="scenario", columns="model", values="margin R²").round(2)
cl = pd.DataFrame(
    {
        s: {
            "frozen": C[s]["model:h3"].mean(),
            "mu-sigma": C[s]["musigma:h3"].mean(),
            "mlp": C[s]["mlp:h3"].mean(),
        }
        for s in ("S1", "S2")
    }
).T
print("margin R² at 4 s (val)")
print(off.to_string())
print("\nwins in the closed loop (S1 = marines, S2 = mixed)")
print(cl.round(3).to_string())
mlp = compare(C, lambda s: "mlp:h3", lambda s: "musigma:h3")[["difference", "low", "high", "read-out"]]
mlp.round(3)

# %% [markdown]
# `planning/37` first read this as "the setup limits planning". `planning/39` corrected it: a perfect model with the same settings beats the best simple rule, so
# **model accuracy limits planning**. Cite 39, not 37.

# %% [markdown]
# ## 6. Where do the models choose differently?
# Share of re-plans in which the model picks the same intent as the perfect model (hold and advance merged in contact; three distinct choices, so chance is about a third).

# %%
D = pd.read_pickle(P / "disagree44.pkl").dropna(
    subset=["oracle"]
)  # each column holds the five candidates' scores
NAMES = ["advance/nearest", "advance/focus", "hold/nearest", "hold/focus", "withdraw/nearest"]
TWIN = np.array([0, 1, 0, 1, 2])  # in contact hold = advance (as experiments/planning/disagree44.py)
best = {k: np.array(D[k].tolist()).argmax(1) for k in ("oracle", "frozen", "musigma")}
ct = D["contact"].to_numpy(bool)
same = {
    k: np.where(ct, TWIN[best[k]] == TWIN[best["oracle"]], best[k] == best["oracle"])
    for k in ("frozen", "musigma")
}
agree = pd.DataFrame(same).groupby(D["scenario"].to_numpy()).mean()
agree.loc["all"] = [same["frozen"].mean(), same["musigma"].mean()]
pick = pd.DataFrame(
    {k: pd.Series(best[k]).map(lambda i: NAMES[i]).value_counts(normalize=True) for k in best}
).fillna(0)
print(agree.round(3).to_string())
print("\nshare of picks")
pick.round(3)

# %% [markdown]
# Also recorded: a richer menu of 27 role combinations adds no room (`planning/49`), and perfect geometry with the model's damage still loses to the best fixed intent in
# four of five scenarios (`planning/51`).

# %% [markdown]
# ## Caveats
# - No LLM, no comparison "with against without a world model": the question itself is untested.
# - The opponent is SMAClite's scripted Red; kite is near-dominant against every Red tried (`planning/41`).
# - The oracle runs copies of the game, including its random stream; `planning/29` checked shared noise.
# - In contact, hold and advance are the same battle in 99% of states: five candidates are three choices.
# - You did not write your own expectation before Phase B, the spread and the compartment runs.
# - The interface for Phase 2 (`docs/world_model.md`) was added on 25 Nov and has not been used in any of these results.

# %% [markdown]
# ## Your answer (to write)
# 1. What may you say about RQ5 *now*, without an LLM?
# 2. Is the gap a failure of the world model, of the action set, or of the test bed (kite)? Which section decides it?
# 3. Given the share captured in section 4 (frozen −13% to 39%, μ, σ −4% to 31%), what would the LLM add that the planner did not already have?
# 4. What result next semester would make you answer RQ5 with "no"?
#
# *Answer:*
