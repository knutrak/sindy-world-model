# ruff: noqa: E501, B018  (markdown tables and display-only cell endings)
# %% [markdown]
# # The final model, its claims and its caveats: A\*-2k, with A, A\*, B and S beside it
#
# **Final model since 19 Nov: A\*-2k** = A's two firepower coefficients k_B, k_R + A\*'s alive law
# (`REPORT.md` §6). A\* (one shared k, the 18 Nov choice) stays in every comparison.
#
# This notebook goes through what the final exploration (`docs/final_exploration/REPORT.md`) found,
# organised as **claims**. Each claim gets one figure or table, and the caveat sits right next to it.
#
# | # | claim | evidence here | caveat beside it |
# |---|---|---|---|
# | 1 | The coefficients are physics | learned vs. game values | p has no "true value" |
# | 2 | What makes A\*-2k differ from A and A\* | alive law without attenuation; why not one shared k | designed after seeing cv / val |
# | 3 | What B and S buy with trajectory fitting | coefficients and margin R² vs. horizon | they read less like physics |
# | 4 | It predicts within a horizon | error vs. horizon for all models | 2-3 s reliable horizon; MLP better everywhere |
# | 5 | What it looks like in practice | three battles chosen by rule | — |
# | 6 | It responds to intents | ranking of the six cf branches | measured on `cf` (used before), not test2 |
# | 7 | It is robust and transfers | gates G1-G8, test2 table, stalkers | gates post-hoc; A\*-2k has no test2 number |
# | 8 | Where it fails | error vs. withdraw share | — |
#
# **Data used:** every model is fitted on exp06 `train` only. Computed sections score on `val` and
# `cf_*`, which were used for choices during the project, so they are *not* independent tests. The
# held-out test2 numbers come from `03_test_results.json` (run once, 18 Nov); test2 is not read here.
#
# **Pre-registered vs. post-hoc**
#
# | what | status | outcome |
# |---|---|---|
# | 17 Nov gates on `final` (A, B, S, W) | pre-registered | B "valid interpretable", S "valid predictive extension", A "not valid" (G3b, alive terms only) |
# | 18 Nov test2: 5 expectations for A\*, B, S | pre-registered | 5 / 5 met |
# | A\* design (shared k, reach-fitted alive law) | post-hoc | chosen after cv / val |
# | 17 Nov gates applied to A\* | post-hoc (val / cf / exp07 val) | passes G1-G8 |
# | switch to A\*-2k (separate k), 19 Nov | post-hoc | ties A\* on cv / val, transfers better; Red reach at 5.375 tested and rejected |
# | 17 Nov gates applied to A\*-2k | post-hoc (val / cf / exp07 val) | passes G1-G6, G8; G7 not evaluated (no test2) |
# | everything computed in this notebook | exploratory | — |

# %%
import json
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display
from scipy.stats import spearmanr

from sindy_wm.evaluation import harness as H
from sindy_wm.evaluation.explore_models import AliveVariantA, CombinedA
from sindy_wm.evaluation.harness_models import ModelA, ModelB, ModelS
from sindy_wm.models.intent_model import DT, make_batch
from sindy_wm.paths import PROJECT_ROOT

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 160)
DOCS = PROJECT_ROOT / "docs" / "final_exploration"
COLORS = {
    "A": "#7f7f7f",
    "A*": "#1f77b4",
    "A*2k": "#d62728",
    "B": "#ff7f0e",
    "S": "#2ca02c",
    "truth": "black",
}
INTERP = ["A", "A*", "A*2k", "B", "S"]  # the interpretable models, in plotting order

# %% [markdown]
# ## 0. The five models
#
# All five share the same state x = [H_B, H_R, A_B, A_R, d], the same controls u = [adv, hold, wd, focus],
# and the same reach closure R = A·Φ((ρ − gap)/spread). They differ in which equations are fitted and how.
#
# | model | health | alive law | distance | fitted on | terms |
# |---|---|---|---|---|---|
# | **A** (`model_derived`) | k_B·R_R, k_R·R_B + k_wd·R_B·wd, STLSQ | (ln A)' = p (ln H)', regress on the smoothed rate ratio | STLSQ, 5 terms | one-step rates (SG 9) | 11 |
# | **A\*** (18 Nov) | **one k for both sides** + k_wd | **A' = c·R_enemy·A/H, p = c/k** | same as A | one-step rates | 10 |
# | **A\*-2k** (final, 19 Nov) | A's k_B, k_R + k_wd | A\*'s, with each side's own k | same as A | one-step rates | 11 |
# | **B** | A's terms | A's terms | A's terms | coefficients refitted on 1-8 s rollout error, + offset decay τ | 11 |
# | **S** | B | B | B | B + the share in reach as a state, rollout-fitted | 22 |
#
# The models are loaded from `data/processed/final_exploration/final_<name>.pkl`, which
# `scripts/build_final_model.py --model <name>` writes (fitted on all of `train`). If one is missing,
# it is fitted here (B and S take about 5 min each).

# %%
FILES = {
    "A": "final_A.pkl",
    "A*": "final_Astar.pkl",
    "A*2k": "final_Astar2k.pkl",
    "B": "final_B.pkl",
    "S": "final_S.pkl",
}
CLASSES = {
    "A": ModelA,
    "A*": CombinedA,
    "A*2k": lambda: AliveVariantA(law="reach"),
    "B": ModelB,
    "S": ModelS,
}
_train = None


def train_data():
    global _train
    if _train is None:
        rates = H.rows_with_rates("train")
        _train = H.FitData(states=H.load_states("train"), rates=H.both_alive(rates), raw_rates=rates)
    return _train


models = {}
for name, f in FILES.items():
    path = H.CACHE / f
    if path.exists():
        models[name] = pickle.load(open(path, "rb"))
    else:
        print(f"fitting {name} (no {f})")
        models[name] = CLASSES[name]().fit(train_data())
        pickle.dump(models[name], open(path, "wb"))
pd.DataFrame({n: m.complexity() for n, m in models.items()})

# %% [markdown]
# ## 1. Claim: the coefficients are physics
#
# All five models can be written in the same parameter vocabulary (the `sindyc_07_lab` dict), so
# their numbers can be put side by side. The scale: health 1 = ten marines × 45 hp; R in tens of units.
#
# Game values:
# - a firing marine deals 6 hp per 0.625 s = 9.6 hp/s, which is **0.213** on this scale per unit in reach,
#   *if every unit in reach fires all the time*;
# - walking speed **3.15** map units/s: closing at 6.3 when both walk (advance), 3.15 when one walks
#   (Blue holds before contact, or withdraws and Red chases);
# - a withdrawing unit does not shoot, so k_wd = +0.213 would cancel all of Blue's fire.


# %%
def params(model) -> dict:
    if hasattr(model, "params"):  # A / A*: no offset decay (the lab dict's tau = 1 is a placeholder)
        return {k: v for k, v in model.params().items() if not k.startswith("tau")}
    return dict(model.p)


P = {n: params(m) for n, m in models.items()}
TRUTH = {
    "kb": -0.213,
    "kr": -0.213,
    "kr_wd": 0.213,
    "d_adv": -6.3,
    "d_wd": -3.15,
    "d_flee": np.nan,
    "d_hold_far": -3.15,
    "d_hold_contact": -6.3,
    "p_b": np.nan,
    "p_r": np.nan,
    "p_r_focus": np.nan,
    "tau_b": np.nan,
    "tau_r": np.nan,
}
MEANING = {
    "kb": "Blue health lost per Red unit in reach",
    "kr": "Red health lost per Blue unit in reach",
    "kr_wd": "fire Blue gives up when withdrawing",
    "d_adv": "closing speed, both advance",
    "d_wd": "closing speed inside the gate, Blue withdraws",
    "d_flee": "opening speed beyond the gate, Blue withdraws",
    "d_hold_far": "closing speed, Blue holds before contact",
    "d_hold_contact": "closing speed, Blue holds in contact",
    "p_b": "alive exponent, Blue",
    "p_r": "alive exponent, Red",
    "p_r_focus": "extra Red exponent under Blue focus fire",
    "tau_b": "offset decay time, Blue (B/S only)",
    "tau_r": "offset decay time, Red (B/S only)",
}
coef = pd.DataFrame({n: {k: P[n].get(k, np.nan) for k in TRUTH} for n in P})
coef.insert(0, "game", pd.Series(TRUTH))
coef["meaning"] = pd.Series(MEANING)
coef.round(3)

# %% [markdown]
# 95% bootstrap ranges of A\*-2k, A\* and A (30 battle-resampled refits each, closures fixed;
# `final_<model>_bootstrap.csv`). The ranges cover data noise only, not the closure
# or the derivative estimate (see the derivative robustness in section 7).

# %%
boot = {}
for name, f in (
    ("A*2k", "final_Astar2k_bootstrap.csv"),
    ("A*", "final_Astar_bootstrap.csv"),
    ("A", "final_A_bootstrap.csv"),
):
    if (DOCS / f).exists():
        boot[name] = pd.read_csv(DOCS / f).set_index("coefficient")[["value", "q2.5", "q97.5"]]
pd.concat(boot, axis=1).round(4) if boot else "no bootstrap tables found"

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
keys = ["kb", "kr", "kr_wd"]
x = np.arange(len(keys))
for j, n in enumerate(INTERP):
    ax.bar(x + (j - 2) * 0.16, [abs(P[n].get(k, np.nan)) for k in keys], 0.16, label=n, color=COLORS[n])
ax.scatter(
    x, [abs(TRUTH[k]) for k in keys], marker="_", s=900, color="k", zorder=5, label="game (all in reach fire)"
)
ax.set_xticks(x, ["|k_B|", "|k_R|", "k_wd"])
ax.set_title("Firepower: |coefficient| vs. the game's rate")
ax.legend(fontsize=8)
ax = axes[1]
keys = ["d_adv", "d_hold_contact", "d_hold_far", "d_wd"]
x = np.arange(len(keys))
for j, n in enumerate(INTERP):
    ax.bar(x + (j - 2) * 0.16, [-P[n].get(k, np.nan) for k in keys], 0.16, label=n, color=COLORS[n])
ax.scatter(
    x, [-TRUTH[k] for k in keys], marker="_", s=900, color="k", zorder=5, label="game (walking speeds)"
)
ax.set_xticks(x, ["advance", "hold, contact", "hold, far", "withdraw"])
ax.set_ylabel("closing speed (map units / s)")
ax.set_title("Distance: closing speeds vs. walking speed")
plt.tight_layout()

# %% [markdown]
# **Reading.** A, A\* and A\*-2k recover the walking speeds within about 5% (6.0 vs 6.3, 3.16-3.18 vs
# 3.15), and their firepower is 83-87% of the "everyone in reach fires" rate, plausible because units in
# reach are sometimes still turning or on cooldown. A and A\*-2k keep k_B < k_R (section 2 explains why);
# A\* forces them equal. B and S give up much of that. Trajectory fitting
# shrinks k (to compensate for the closure's early, full contact), adds offset decays τ that have no
# game counterpart, and moves the closing speeds away from the walking speeds (advance 7.7 instead of
# 6.3; hold-far and withdraw about 2.0-2.2 instead of 3.15). S refits only its share terms, so its
# health, distance and alive numbers are B's.
#
# **Caveat:** p (the alive exponents) has no game value to compare with. It summarises how damage spreads
# over units before one dies, which depends on targeting.
#
# *To think about:* is "about 85% of the game rate" evidence for physics, or a fitted efficiency? What
# would you need to see to tell those apart?

# %% [markdown]
# ## 2. What makes A\*-2k differ from A, and why not A\*'s one k
#
# The distance equation is the same in all three. A\*-2k changes only the alive law (point 2). A\* also
# changed the health equations (point 1), and that change was taken back on 19 Nov.
#
# 1. **One firepower coefficient for both sides? No.** The units are identical, but the controllers are
#    not: SMAClite's Red AI picks up a new target only within its scan range + the target's radius
#    (5 + 0.375 = 5.375), while Blue shoots out to 5.75. Red keeps targets out to 5.75 and fights back
#    against units shooting it. So some Red units counted in R_R do not fire yet, and k_B (Blue's loss per
#    Red unit in reach) should be lower than k_R. A's separate fit shows exactly that: 0.177 vs 0.186,
#    with bootstrap ranges that do not overlap. A\* fits one k on both equations stacked and hides it.
# 2. **The alive law fitted without dividing by a noisy rate.** A regresses (ln A)' on (ln H)'. Both are
#    ratios of smoothed derivatives, so the regressor is noisy, and least squares with a noisy
#    regressor pulls the slope towards 0 (errors-in-variables attenuation). In a rollout, H' = k·R_enemy,
#    so the law the rollout actually uses is A' = p·k·R_enemy·A/H. A\* regresses A' on R_enemy·A/H,
#    a regressor built from states only, and gets p = c/k.

# %%
r = train_data().rates
rows = []
for side, enemy in (("blue", "red"), ("red", "blue")):
    ok = ((r[f"health_{side}"] > 0.03) & (r[f"alive_{side}"] > 0)).to_numpy()
    g = (r[f"d_health_{side}"] / r[f"health_{side}"]).to_numpy()[ok]
    y = (r[f"d_alive_{side}"] / r[f"alive_{side}"]).to_numpy()[ok]
    x = (r[f"reach_{enemy}"] * r[f"alive_{side}"] / r[f"health_{side}"]).to_numpy()[ok]
    ya = r[f"d_alive_{side}"].to_numpy()[ok]
    k = P["A*2k"]["kb" if side == "blue" else "kr"]  # each side's own firepower
    p_ratio = float(np.dot(g, y) / np.dot(g, g))
    p_reach = float(np.dot(x, ya) / np.dot(x, x) / k)
    rows.append(
        {"side": side, "p from rate ratio (A's regressor)": p_ratio, "p from R·A/H (A*-2k's)": p_reach}
    )
pd.DataFrame(rows).set_index("side").round(3)

# %% [markdown]
# (Without the focus term, so the Red numbers differ slightly from the fitted models'.)
#
# The figure shows why: binned means of the alive rate against what each law predicts it to be. A
# law with the right p puts the points on the diagonal.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, side, enemy, p_key in ((axes[0], "blue", "red", "p_b"), (axes[1], "red", "blue", "p_r")):
    ok = ((r[f"health_{side}"] > 0.03) & (r[f"alive_{side}"] > 0) & (r["focus"] == 0)).to_numpy()
    base = (P["A*"]["kb"] * r[f"reach_{enemy}"] * r[f"alive_{side}"] / r[f"health_{side}"]).to_numpy()[ok]
    ya = r[f"d_alive_{side}"].to_numpy()[ok]
    bins = np.quantile(base, np.linspace(0, 1, 26))
    idx = np.clip(np.digitize(base, bins) - 1, 0, 24)
    bx = np.array([base[idx == i].mean() for i in range(25)])
    by = np.array([ya[idx == i].mean() for i in range(25)])
    ax.scatter(bx, by, s=14, color="k", label="data (binned means)")
    for n in ("A", "A*", "A*2k"):
        p = P[n][p_key]
        kk = P[n]["kb" if side == "blue" else "kr"] / P["A*"]["kb"]  # the model's own k for that side
        ax.plot(bx, p * kk * bx, color=COLORS[n], label=f"{n}: p = {p:.3f}")
    ax.set_xlabel("k·R_enemy·A/H  (alive rate if p = 1)")
    ax.set_ylabel(f"A_{side}' (alive rate)")
    ax.set_title(f"{side}: the alive law along the rollout's own regressor")
    ax.legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# **What it changes in prediction** (val and stalkers; from `results.csv`, the same harness as everywhere
# else; A\*-2k is the row `G/alive-reach`). The alive law gives small gains over A (on cv A\* beat A in
# 5 / 5 folds on health 4 s, alive 4 s and battle nRMSE; on test2, run once, health 4 s went from 0.054 to
# 0.052, [−0.0023, −0.0016]). A\*-2k ties A\* on val. On stalkers (scaled, no refit) A\*-2k keeps A's
# margin transfer, which A\*'s shared k loses.

# %%
res = H.load_results()
NAMES = {
    "ref/A": "A",
    "deep/A*": "A*",
    "G/alive-reach": "A*2k",
    "ref/B": "B",
    "ref/S": "S",
    "J/mlp-wide": "MLP",
    "ref/persistence": "persistence",
    "ref/lanchester-square": "Lanchester",
    "ref/no-control": "no control",
}
val = res[(res.split == "val") & res.experiment.isin(NAMES)].drop_duplicates("experiment", keep="last")
val = val.assign(model=val.experiment.map(NAMES)).set_index("model")
cols = [
    "health_4s",
    "alive_4s",
    "distance_4s",
    "margin_r2_2s",
    "margin_r2_4s",
    "battle_health_nrmse",
    "n_terms",
]
stalk = res[(res.split == "stalkers") & res.experiment.isin(NAMES)].drop_duplicates("experiment", keep="last")
stalk = stalk.assign(model=stalk.experiment.map(NAMES)).set_index("model")
pd.concat(
    {
        "val": val.loc[INTERP, cols],
        "stalkers": stalk.loc[INTERP, ["health_4s", "margin_r2_4s", "battle_health_nrmse"]],
    },
    axis=1,
).round(3)

# %% [markdown]
# **Robustness is where the new alive law matters most.** A failed the pre-registered G3b gate because
# its alive term set changed with the derivative estimate. The reach-regressor alive law (A\*, A\*-2k) has
# no term selection, so its term set cannot change (partly by design). What can still move is the numbers:

# %%
gates = json.loads((DOCS / "04_gates_post_hoc.json").read_text())
gates["A*2k"] = json.loads((DOCS / "05_gates_separate_k.json").read_text())["A*, separate k"]
g3 = []
for n in ("A", "A*", "A*2k"):
    d = gates[n]["G3b detail"]
    for term, v in d.get("health coefficients", {}).items():
        g3.append(
            {
                "model": n,
                "coefficient": term,
                "values (SG 5/9/17, weak 1/2 s)": v["values"],
                "max dev. from median": round(v["max deviation from median"], 3),
            }
        )
    g3.append(
        {
            "model": n,
            "coefficient": "same term set",
            "values (SG 5/9/17, weak 1/2 s)": d.get("same term set"),
            "max dev. from median": None,
        }
    )
pd.DataFrame(g3)

# %% [markdown]
# *Caveat:* A\* and A\*-2k were designed after looking at cv and val, so their gate passes are post-hoc
# support, not a pre-registered confirmation. A's failure reproduces exactly (same term-set changes at SG 17 and weak
# 1 s / 2 s), so the procedure itself is not the difference.

# %% [markdown]
# ## 3. B and S: what trajectory fitting buys, and what it costs
#
# B keeps A's 11 terms and refits all coefficients plus two offset decay times on 1-8 s rollout error.
# S adds the share of units in reach as a state with its own 11 terms. Starting B from A\* instead of A
# gives identical coefficients (B\* = B), so the rollout fit has one optimum.

# %%
show = ["kb", "kr", "kr_wd", "p_b", "p_r", "p_r_focus", "tau_b", "tau_r"]
cmp = coef.loc[show, ["game", *INTERP, "meaning"]].copy()
cmp["B / A*-2k"] = coef.loc[show, "B"] / coef.loc[show, "A*2k"]
cmp.round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
hs = [1, 2, 4, 8]
for ax, metric, title in (
    (axes[0], "health_{}s", "health error (|ΔH|)"),
    (axes[1], "margin_r2_{}s", "margin R² (who gains)"),
    (axes[2], "alive_{}s", "alive error (units)"),
):
    for n in INTERP:
        ax.plot(hs, [val.loc[n, metric.format(h)] for h in hs], "o-", color=COLORS[n], label=n)
    ax.set_xscale("log", base=2)
    ax.set_xticks(hs, [f"{h} s" for h in hs])
    ax.set_title(title)
axes[1].axhline(0, color="k", lw=0.5)
axes[0].legend()
plt.suptitle("val: the interpretable models against horizon", y=1.02)
plt.tight_layout()

# %% [markdown]
# **Reading.** The trajectory-fitted models gain mainly on the **margin** (who is winning) at 2-4 s:
# margin R² 4 s about 0.28 (B) and 0.41 (S) against about 0.1 for A / A\* / A\*-2k. Health error moves less. They
# pay for it with smaller firepower coefficients, offset decays, and (for S) two Red share terms with
# the wrong sign (G2). Over whole battles S is *worse* than A (battle nRMSE 0.141 vs 0.131).
#
# *To think about:* the margin is what a planner needs. Does that make S the model for RQ3 and A\*-2k the
# model for the "physics" RQ, or is one model per thesis cleaner?

# %% [markdown]
# ## 4. Claim: it predicts well within a horizon
#
# All models on val, including the black-box MLP (19 205 weights) as the upper reference and three
# baselines. This is the figure that shows the price of interpretability in numbers.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
order = ["persistence", "Lanchester", "no control", *INTERP, "MLP"]
style = {"persistence": ":", "Lanchester": ":", "no control": ":", "MLP": "--"}
extra = {"persistence": "#bbbbbb", "Lanchester": "#8c564b", "no control": "#e377c2", "MLP": "#9467bd"}
for ax, metric, title in ((axes[0], "health_{}s", "health error"), (axes[1], "margin_r2_{}s", "margin R²")):
    for n in order:
        if n in val.index:
            ax.plot(
                hs,
                [val.loc[n, metric.format(h)] for h in hs],
                "o" + style.get(n, "-"),
                color=COLORS.get(n, extra.get(n)),
                label=n,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(hs, [f"{h} s" for h in hs])
    ax.set_title(f"val: {title}")
axes[1].set_ylim(-0.6, 0.8)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].axvspan(1, 2.5, color="#1f77b4", alpha=0.06)
axes[1].text(1.05, -0.5, "A*-2k reliable\nhorizon (G6)", fontsize=8)
axes[0].legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# The Pareto view (cv, from the harness): complexity against health error at 4 s.

# %%
display(Image(filename=str(DOCS / "pareto_cv.png"), width=650))

# %% [markdown]
# **Caveat:** the reliable horizon (G6: margin R² interval above a threshold) is 2 s on val and 3 s on
# cf, the same for A, A\* and A\*-2k. The MLP is better at every horizon; the gap at 4 s (health 0.033 vs
# 0.053) is the price of an 11-term readable model.

# %% [markdown]
# ## 5. What it looks like: three battles, chosen by rule
#
# Closed-loop rollouts from step 0 to the end of the battle under the real intent sequence (the harness's
# battle rollout). The battles are chosen by A\*-2k's whole-battle error: the **median** battle, the
# **90th percentile**, and the median among **withdraw-heavy** battles (Blue withdraws > 35% of the time).

# %%
val_states = H.load_states("val")
Bb = H.battle_batch(val_states)
Yb = {n: m.rollout(Bb) for n, m in models.items()}


def per_battle_nrmse(Y, B):
    Yt, end = B["Y"], B["end"]
    start = np.maximum(Yt[:, 0, :2], 1e-6)
    mask = np.arange(Yt.shape[1])[None, :] <= end[:, None]
    err = np.where(mask[:, :, None], (Y[:, :, :2] - Yt[:, :, :2]) / start[:, None, :], 0.0)
    return np.sqrt((err**2).sum((1, 2)) / (2 * (end + 1)))


nr = pd.DataFrame({n: per_battle_nrmse(Y, Bb) for n, Y in Yb.items()})
wd_share = val_states.groupby(level=0)["withdraw"].mean().loc[Bb["ep"]].to_numpy()
nr["withdraw share"] = wd_share
nr.describe(percentiles=[0.5, 0.9]).loc[["mean", "50%", "90%"]].round(3)


# %%
def pick(values, q, mask=None):
    v = np.where(np.ones(len(values), bool) if mask is None else mask, values, np.nan)
    target = np.nanquantile(v, q)
    return int(np.nanargmin(np.abs(v - target)))


picks = {
    "median battle": pick(nr["A*2k"].to_numpy(), 0.5),
    "90th percentile": pick(nr["A*2k"].to_numpy(), 0.9),
    "withdraw-heavy (median)": pick(nr["A*2k"].to_numpy(), 0.5, wd_share > 0.35),
}
fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex="row")
for row, (label, i) in enumerate(picks.items()):
    end = Bb["end"][i]
    t = np.arange(end + 1) * DT
    tm = np.arange(min(end + 24, Bb["Y"].shape[1])) * DT
    for col, (j, ylab) in enumerate(((0, "health Blue"), (1, "health Red"), (4, "distance"))):
        ax = axes[row, col]
        ax.plot(t, Bb["Y"][i, : end + 1, j], color="k", lw=2, label="truth")
        for n in INTERP:
            ax.plot(tm, Yb[n][i, : len(tm), j], color=COLORS[n], lw=1.2, label=n)
        ax.axvline(end * DT, color="k", lw=0.5, ls=":")
        ax.set_ylabel(ylab)
        if col == 0:
            ax.set_title(
                f"{label}: battle {Bb['ep'][i]}, A*-2k nRMSE {nr['A*2k'][i]:.3f}, withdraw {wd_share[i]:.0%}",
                loc="left",
                fontsize=9,
            )
    u = Bb["U"][i, : end + 1]
    for k, c in ((1, "#ffdd99"), (2, "#ffbbbb")):
        on = u[:, k] > 0.5
        for ax in axes[row]:
            ax.fill_between(t, 0, 1, where=on, color=c, alpha=0.5, transform=ax.get_xaxis_transform(), lw=0)
axes[0, 0].legend(fontsize=7)
axes[2, 0].set_xlabel("s")
plt.tight_layout()

# %% [markdown]
# Shading: yellow = Blue holds, red = Blue withdraws. The dotted line is the real end of the battle.

# %% [markdown]
# ## 6. Claim: it responds correctly to intents (counterfactual ranking)
#
# This is what a planner needs. In `cf_*` the same battle is replayed six times: identical up to a drawn
# branch moment, then one of six intents for 4 s, then advance/nearest. For each seed, we roll every
# model out from the branch moment under each branch's real controls and ask:
#
# - **Does the choice matter?** How much do the true outcomes spread across the six branches?
# - **Does the model rank them like the simulator?** Spearman rank correlation across the six,
#   how often the model's best is the true best, and the regret (true best − true value of the model's pick).
#
# Outcome: the change in the health margin (H_B − H_R) at 4 s (the end of the branch window) and at
# 8 s, or at the battle's end if earlier. *Caveat:* `cf_*` was used during the project (and for the
# 17 Nov dry runs), so this is exploratory.


# %%
def cf_batch(states, horizon=64):
    parts = []
    for _, g in states.groupby(level=0, sort=False):
        k = int(round(g["branch_at"].iloc[0] / DT))
        if k < len(g) - 8 and g["alive_blue"].iloc[k] > 0 and g["alive_red"].iloc[k] > 0:
            parts.append(g.iloc[k:])
    sub = pd.concat(parts)
    with np.errstate(invalid="ignore", divide="ignore"):
        B = make_batch(sub, horizon=horizon, every=10**9, history=0, min_future=1)
    B["seed"], B["branch"] = B["ep"] % 10_000, B["ep"] // 10_000 - 1
    return B


cf_states = H.load_states("cf")
LABELS = sorted(p.stem.removeprefix("cf_") for p in (H.PROCESSED / "exp06_intents").glob("cf_*.parquet"))
Bc = cf_batch(cf_states)
Yc = {n: m.rollout(Bc) for n, m in models.items()}
print(f"{len(Bc['X0'])} branch starts, {len(np.unique(Bc['seed']))} seeds; branches: {LABELS}")


# %%
def margin_change(Y, B, h):
    """Margin change at h seconds, or at the battle's real end if that is earlier."""
    n = round(h / DT)
    valid = ~np.isnan(B["Y"][:, :, 0])
    last = valid.sum(1) - 1
    idx = np.minimum(n, last)
    rows = np.arange(len(Y))
    return (Y[rows, idx, 0] - Y[rows, idx, 1]) - (Y[:, 0, 0] - Y[:, 0, 1])


def ranking(h):
    true = margin_change(Bc["Y"], Bc, h)
    frame = pd.DataFrame(
        {"seed": Bc["seed"], "branch": Bc["branch"], "truth": true}
        | {n: margin_change(Y, Bc, h) for n, Y in Yc.items()}
    )
    wide = {c: frame.pivot(index="seed", columns="branch", values=c) for c in ["truth", *Yc]}
    full = wide["truth"].dropna().index
    T = wide["truth"].loc[full].to_numpy()
    out = {"spread across branches (std of true margin change)": T.std(1).mean()}
    table = {}
    for n in Yc:
        M = wide[n].loc[full].to_numpy()
        rho = np.nanmean([spearmanr(M[i], T[i]).statistic for i in range(len(T))])
        pick_ = M.argmax(1)
        table[n] = {
            "rank correlation": rho,
            "picks the true best": float((T[np.arange(len(T)), pick_] >= T.max(1) - 1e-9).mean()),
            "regret": float((T.max(1) - T[np.arange(len(T)), pick_]).mean()),
        }
    best_fixed = T.mean(0).argmax()
    table[f"always {LABELS[best_fixed]}"] = {
        "rank correlation": np.nan,
        "picks the true best": float((T[:, best_fixed] >= T.max(1) - 1e-9).mean()),
        "regret": float((T.max(1) - T[:, best_fixed]).mean()),
    }
    rng = np.random.default_rng(0)
    rnd = rng.integers(0, T.shape[1], len(T))
    table["random"] = {
        "rank correlation": 0.0,
        "picks the true best": float((T[np.arange(len(T)), rnd] >= T.max(1) - 1e-9).mean()),
        "regret": float((T.max(1) - T[np.arange(len(T)), rnd]).mean()),
    }
    return out | {"seeds": len(full)}, pd.DataFrame(table).T


for h in (4, 8):
    info, table = ranking(h)
    display(
        Markdown(
            f"**At {h} s** — {info['seeds']} seeds; mean spread of the true margin change "
            f"across the six branches: {info['spread across branches (std of true margin change)']:.3f}"
        )
    )
    display(table.round(3))

# %% [markdown]
# **How to read it.** "Picks the true best" for random is about 1/3, not 1/6, because branches often
# tie (for example focus and nearest before contact). The "always <intent>" row is the best *fixed* intent,
# chosen in hindsight on these same seeds, so it is slightly optimistic. Regret is in health units
# (1 = ten marines): the margin a planner would give away by trusting the model.
#
# **Observation (19 Nov run):** A\*-2k ranks like A\* (rank correlation 0.52 at 4 s, 0.21 at 8 s). At
# 4 s all five models rank the branches with a rank correlation of
# about 0.5 and pick the true best about as often as the best fixed intent (0.45-0.49 vs 0.46), with a
# slightly *higher* regret (0.030-0.032 vs 0.027). At 8 s the correlation drops to about 0.2 and none
# beats the fixed rule. Model differences are small here; S is the only one above the fixed rule on
# "picks the best" at 4 s.
#
# *To think about:* what does it mean for a planner claim that a model ranks clearly better than random
# but not better than "always advance/nearest"? Is the spread across branches large enough that ranking
# matters at all? And would a decision every 1-2 s (inside the reliable horizon) change this?

# %% [markdown]
# ## 7. Claim: robust and transfers (gates and the held-out test)
#
# The 17 Nov gates applied post-hoc to A\* and, as a control, to A (`04_gates_post_hoc.md`), and to
# A\*-2k (`05_gates_separate_k.md`, 19 Nov); and the pre-registered test2 run (`03_test_results.md`, run
# once on 18 Nov). A\*-2k was chosen after test2 was spent, so it has no test2 row and no G7.

# %%
gate_rows = {}
for n in ("A", "A*", "A*2k"):
    g = gates[n]
    gate_rows[n] = {k: g.get(k) for k in ("G1", "G2", "G3a", "G3b", "G4", "G5", "G7", "G8")} | {
        "horizon val / cf (s)": f"{g['G6']['val']} / {g['G6']['cf']}",
        "verdict": g["verdict"],
        "transfers": g["transfers"],
    }
pd.DataFrame(gate_rows).T

# %%
t2 = json.loads((DOCS / "03_test_results.json").read_text())
tab = {}
for split in ("test2", "test2_cf", "test2_stalkers"):
    for exp, m in t2["splits"][split]["metrics"].items():
        tab[(split, NAMES.get(exp, exp))] = {
            c: m.get(c) for c in ("health_4s", "margin_r2_4s", "battle_health_nrmse")
        }
t2_table = pd.DataFrame(tab).T.unstack(0).swaplevel(0, 1, axis=1).sort_index(axis=1)
t2_table.loc[
    [
        n
        for n in ["A", "A*", "B", "S", "MLP", "no control", "Lanchester", "persistence"]
        if n in t2_table.index
    ]
].round(3)

# %%
ci = pd.DataFrame(
    {
        NAMES.get(k, k): {m: f"[{v[0]:+.4f}, {v[1]:+.4f}]" for m, v in d.items()}
        for k, d in t2["splits"]["test2"]["minus_A_95"].items()
    }
).T
display(
    Markdown(
        "**test2: paired 95% intervals of model − A** (below 0 is better for health and nRMSE, "
        "above 0 for margin R²)"
    )
)
ci

# %% [markdown]
# **Caveats.** The gate passes for A\* and A\*-2k are post-hoc on sets used before. The bootstrap
# intervals understate uncertainty (data noise only). On stalkers, A\* transfers the margin *worse* than A
# (expectation 4 in the pre-registration, met). A\*-2k, with the same alive law but separate k, does not
# (section 2), so the loss came mostly from the shared k.

# %% [markdown]
# ## 8. Where it fails: withdraw-heavy battles
#
# Whole-battle error against the share of the battle Blue spends withdrawing (val), and the 4 s
# distance and health errors against the withdraw share inside the 4 s window (start points). If the
# problem is the *state* (one distance cannot describe a stretched-out formation), distance error should
# grow with the withdraw share. If it is the fire reduction, health error should.

# %%
bins = [-0.01, 0.0, 0.15, 0.35, 1.0]
labels = ["none", "0-15%", "15-35%", "> 35%"]
nr["withdraw bin"] = pd.cut(nr["withdraw share"], bins, labels=labels)
by_bin = nr.groupby("withdraw bin", observed=True)[INTERP].mean()
by_bin["battles"] = nr.groupby("withdraw bin", observed=True).size()

Bh = H.horizon_batch(val_states)
n4 = round(4 / DT)
ok = ~np.isnan(Bh["Y"][:, n4, 0])
wd_win = Bh["U"][:, :n4, 2].mean(1)
start_bin = pd.Categorical(pd.cut(wd_win, bins, labels=labels), categories=labels, ordered=True)
Yh = {n: m.rollout(Bh) for n, m in models.items()}
err = {}
for n, Y in Yh.items():
    err[(n, "distance 4 s")] = np.abs(Y[:, n4, 4] - Bh["Y"][:, n4, 4])
    err[(n, "health 4 s")] = np.abs(Y[:, n4, :2] - Bh["Y"][:, n4, :2]).mean(1)
# for scale: how much distance and health really change in the window (larger moves, larger errors)
err[("truth", "|Δ distance| 4 s")] = np.abs(Bh["Y"][:, n4, 4] - Bh["Y"][:, 0, 4])
err[("truth", "|Δ health| 4 s")] = np.abs(Bh["Y"][:, n4, :2] - Bh["Y"][:, 0, :2]).mean(1)
E = pd.DataFrame(err)[ok]
E["bin"] = start_bin[ok]
win_table = E.groupby("bin", observed=True).mean()
win_table[("n", "starts")] = E.groupby("bin", observed=True).size()

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for n in INTERP:
    axes[0].plot(labels, by_bin[n].reindex(labels), "o-", color=COLORS[n], label=n)
    axes[1].plot(labels, win_table[(n, "distance 4 s")].reindex(labels), "o-", color=COLORS[n], label=n)
    axes[2].plot(labels, win_table[(n, "health 4 s")].reindex(labels), "o-", color=COLORS[n], label=n)
axes[0].set_title("whole-battle health nRMSE\nvs. share of battle withdrawing")
axes[1].set_title("distance error at 4 s\nvs. withdraw share in the window")
axes[2].set_title("health error at 4 s\nvs. withdraw share in the window")
axes[0].legend()
plt.tight_layout()
display(by_bin.round(3))
win_table.round(3)

# %% [markdown]
# The `truth` columns give the scale: if the real distance moves much more during withdraw, a larger
# absolute error is partly expected. Compare error to movement, not error alone.
#
# *To think about:* which of the two errors grows faster with withdraw? If it is distance, what is the
# smallest state change that would describe a withdrawing formation (the depth / width columns are
# already in the data)? And does removing withdraw fix the model, or only the score?

# %% [markdown]
# ## Summary
#
# | claim | supported? | strength |
# |---|---|---|
# | Coefficients are physics (A, A\*-2k) | speeds within ~5%, firepower 83-87% of the game rate, k_B < k_R as Red's scan radius predicts | good for health and distance; p has no reference |
# | A\*-2k improves on A | alive law: small, consistent gains (A\*: 5/5 folds, test2 interval below 0); passes G3b | post-hoc design and gates; no test2 number for A\*-2k itself |
# | One shared k (A\*) | ties A\*-2k, but hides the asymmetry and loses margin transfer to stalkers | Red reach at 5.375 overcorrects (rejected) |
# | Trajectory fitting (B, S) | better margin at 2-4 s | coefficients read less like physics; S worse over whole battles |
# | Predicts within a horizon | 2 s (val) / 3 s (cf) | MLP better everywhere |
# | Responds to intents | see section 6 | exploratory, on `cf` |
# | Fails on withdraw-heavy battles | see section 8 | — |
