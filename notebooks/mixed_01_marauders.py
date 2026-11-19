# ruff: noqa: E501, B018  (markdown tables and display-only cell endings)
# %% [markdown]
# # Marines and marauders: what changes when a battle has two unit types
#
# Branch `mixed-marauders` (19 Nov). The marine model (A\*-2k, `notebooks/final_01_model_showcase`) is the
# starting point. Two data sets: **exp09** (marauders only) and **exp08** (marines + marauders, the same mix
# on both sides, marauder share ¼, ⅓ or ½). Everything here is exploratory, on `val` and `cf`; the held-out
# sets (exp09 `test`, exp08 `test3`) have not been read. Log: `docs/mixed/PROGRESS.md`.
#
# | # | claim | evidence | caveat |
# |---|---|---|---|
# | 1 | The per-type damage matrix matches the game's pattern | fitted vs. game, 2 × 2 per side | needs measured target choice to fit |
# | 2 | The share of units in reach that fire is not unit-free | marines ≈ 0.8, marauders ≈ 0.54 alone, both higher in mixed battles | why marauders fire less is not known |
# | 3 | The marine model's *structure* transfers to marauders, its numbers don't | refit finds the same 11 terms; scaled transfer gets the margin wrong | — |
# | 4 | The per-type state pays off in rollouts | beats A\*-2k on team totals | alive law per type weak |
# | 5 | What a battle looks like per type | one median battle | — |
# | 6 | Target choice is readable | fire shifts to the type further forward; Red over-targets marauders | fitted closure ≈ proportional rule in rollouts |
# | 7 | One distance is defensible on average | types stay level on average | per-battle spread ≈ formation depth |
# | 8 | Ranking intents | above random, not above the best fixed intent | same as marines |

# %%
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from sindy_wm.evaluation import harness as H
from sindy_wm.models.intent_model import DT, add_controls
from sindy_wm.models.switch_library import scaled
from sindy_wm.models.typed_model import (
    SIDES,
    TYPES,
    idx,
    team_battle_batch,
    team_horizon_batch,
    team_view,
    typed_batch,
)
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT, RAW_DIR
from sindy_wm.states.typed import load_typed_states

sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "mixed"))
import cf_ranking  # noqa: E402
from oracle import game_rate  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180)
P8 = DATA_DIR / "processed" / "exp08_mixed"
COLORS = {"team A*-2k": "#7f7f7f", "typed (closure)": "#d62728", "typed (rule)": "#ff9896", "truth": "black"}
mixed = pickle.load(open(H.CACHE / "mixed_models.pkl", "rb"))  # team A*-2k, typed (closure), typed (rule)
typed = mixed["typed (closure)"]

# %% [markdown]
# ## 0. What changed from the marine model
#
# | | marine A\*-2k | mixed per-type model (`TypedA`) |
# |---|---|---|
# | state | H_B, H_R, A_B, A_R, d | H and A **per type** and side, plus d (9) |
# | health | H_B' = k_B·R_R | **2 × 2 matrix per side**: Σ_a k(a→t)·R_a·σ(a→t) |
# | target choice | — | **closure σ**: share of type a's fire on marauders, logistic in the target side's marauder share and front offset |
# | reach | one closure per side, ρ = 5.75 | **per type**, ρ = the type's reach (≈ 5.84 marines, ≈ 7.03 marauders), shifted by how far forward the type sits |
# | withdraw | k_wd on Blue's fire | per attacker type (kept for Blue's marauders) |
# | alive | p per side | p per type and side |
# | distance | marine gates | the same equation, refitted on mixed data |
# | unchanged | | one-step STLSQ, A\*-2k's alive-law form, offset-decaying reach, team-wide intents, shape held |
#
# The groups are unit *types*, not spatial clusters: units are shuffled within the formation and share one
# centroid distance.

# %% [markdown]
# ## 1. Claim: the damage matrix matches the game's pattern
#
# Fitted (one step, measured reach × measured share of fire) against the game's rate per *firing* unit:
# (damage + bonus − armor) / cooldown rounded up to ticks / target hp. Units: target units removed per
# second. The fitted value is per unit *in reach*, so it is the game rate times the share of units in reach
# that fire (claim 2).

# %%
rows = []
for s in SIDES:
    o = "red" if s == "blue" else "blue"
    for a in TYPES:
        for t in TYPES:
            k = -typed.coef[(s, t)].get(f"R[{a}]*s[{t}]", 0.0)
            g = game_rate(a, t)
            rows.append(
                {"attacker": f"{a} ({o})", "target": t, "game": g, "fitted": k, "fitted / game": k / g}
            )
matrix = pd.DataFrame(rows).set_index(["attacker", "target"])
display(matrix.round(3))

fig, ax = plt.subplots(figsize=(9, 3.6))
lab = [
    f"{a[:3]}→{t[:3]}\n({o})"
    for (a, t, o) in [
        (r["attacker"].split()[0], r["target"], r["attacker"].split()[1])
        for _, r in matrix.reset_index().iterrows()
    ]
]
x = np.arange(len(matrix))
ax.bar(x - 0.2, matrix["game"], 0.4, label="game, per firing unit", color="#bbbbbb")
ax.bar(x + 0.2, matrix["fitted"], 0.4, label="fitted, per unit in reach", color="#d62728")
ax.set_xticks(x, lab, fontsize=8)
ax.set_ylabel("target units / s")
ax.legend()
plt.tight_layout()

# %% [markdown]
# **Reading.** The pattern is the game's: marines barely hurt marauders (0.053-0.055 against 0.064), and the
# marauder's +10 against armored shows in marauder → marauder. Every entry is below the game rate, by the
# share of units in reach that are not firing. Blue (scripted) is more effective per unit in reach than Red
# (AI) for both attacker types.
#
# **Caveat** (checkpoint 2): the rate *per firing unit* is not identifiable when reach is also in the
# library, because firing ≈ 0.8 × reach. Which of two collinear descriptions STLSQ keeps depends on the library.

# %% [markdown]
# ## 2. Claim: the share of units in reach that fire depends on the unit type and the battle
#
# Firing / in reach, from the snapshots (marines alone: exp06 `val`; marauders alone: exp09 `val`; 200
# battles each) and from the mixed `val` table.

# %%
share_rows = []
for label, exp, t in (
    ("marines alone", "exp06_intents", "MARINE"),
    ("marauders alone", "exp09_marauders", "MARAUDER"),
):
    s = load_typed_states(RAW_DIR / exp / "val", (t,), episode_ids=list(range(200)))
    tl = t.lower()
    for side in SIDES:
        ok = s[f"reach_{side}_{tl}"] > 0
        share_rows.append(
            {
                "battles": label,
                "side": side,
                "type": tl,
                "firing / in reach": s.loc[ok, f"fire_{side}_{tl}_{tl}"].sum()
                / s.loc[ok, f"reach_{side}_{tl}"].sum(),
            }
        )
val8 = pd.read_parquet(P8 / "val.parquet")
for side in SIDES:
    for a in TYPES:
        ok = val8[f"reach_{side}_{a}"] > 0
        fire = val8[f"fire_{side}_{a}_marine"] + val8[f"fire_{side}_{a}_marauder"]
        share_rows.append(
            {
                "battles": "mixed",
                "side": side,
                "type": a,
                "firing / in reach": fire[ok].sum() / val8.loc[ok, f"reach_{side}_{a}"].sum(),
            }
        )
pd.DataFrame(share_rows).set_index(["battles", "side", "type"]).round(3)

# %% [markdown]
# **Reading.** Marines fire about 80% of the time they are in reach (82-86% in mixed battles), marauders
# alone only 53-54% (58-60% in mixed battles). So k (per unit in reach) is a game rule *times* a behavioural share that
# changes with unit type and with the battle. This is the main limit to reading k as "physics". Not yet
# known: why marauders in reach fire less (bigger bodies crowding, the wider band between Red's acquisition
# radius 5.56 and the reach 7.1, retargeting after kills?).

# %% [markdown]
# ## 3. Claim: the marine model's structure transfers to marauders, its numbers don't
#
# Marauders alone (exp09 `val`, 500 battles). The marine A\*-2k scaled to marauders without refitting
# (`switch_library.scaled`: reach, walking speed, time to kill), against A\*-2k's recipe refitted on exp09 `train`.

# %%
val9 = H.load_states("val", "exp09_marauders")
B9h, B9b = H.horizon_batch(val9), H.battle_batch(val9)
marine = pickle.load(open(H.CACHE / "final_Astar2k.pkl", "rb"))
refit = pickle.load(open(H.CACHE / "marauders_Astar2k.pkl", "rb"))


def roll9(m, B):
    return m.simulate(B["X0"], B["shape0"], B["U"], reach0=B["reach0"])


m_scaled = scaled(marine.m, "marine", "marauder")
r9 = {}
for name, m in (("marine A*-2k, scaled", m_scaled), ("A*-2k refit on marauders", refit.m)):
    r9[name] = {**H.horizon_metrics(roll9(m, B9h), B9h), **H.battle_metrics(roll9(m, B9b), B9b)}
display(
    pd.DataFrame(r9)
    .T[
        [
            "health_2s",
            "health_4s",
            "margin_r2_2s",
            "margin_r2_4s",
            "alive_4s",
            "distance_4s",
            "battle_health_nrmse",
        ]
    ]
    .round(3)
)


def terms(m):
    t = {
        f"{eq}: {n}": c
        for eq in ("health_blue", "health_red", "distance")
        for n, c in zip(m.names, m.coef[eq], strict=True)
        if c
    }
    t |= {f"alive_{s}: {k}": v for s in SIDES for k, v in m.p[s].items() if v}
    return t


pd.DataFrame(
    {
        "marine A*-2k": terms(marine.m),
        "marine, scaled to marauders": terms(m_scaled),
        "refit on marauders": terms(refit.m),
    }
).round(3)

# %% [markdown]
# **Reading.** Same 11 terms, same signs, similar alive exponents. The scaled firepower (≈ 0.12) is too high
# against the refit's 0.077 / 0.086, which is what claim 2 predicts (0.135 × ≈ 0.54). Walking speeds are
# recovered (3.0-3.05 against 3.15); holding in contact closes slower than for marines.

# %% [markdown]
# ## 4. Claim: the per-type state pays off in rollouts
#
# Mixed `val` (1000 battles). Team-level metrics in ten marines' hp, comparable with the marine-only numbers.

# %%
val8c = add_controls(val8)
Bh = typed_batch(val8c)
Th = team_horizon_batch(Bh)
Bb, Tb = team_battle_batch(val8c)
Yh, Yb, Yh_typed = {}, {}, {}
for name, m in mixed.items():
    if name.startswith("typed"):
        Yh_typed[name] = m.rollout(Bh)
        Yh[name], Yb[name] = team_view(Yh_typed[name]), team_view(m.rollout(Bb))
    else:
        Yh[name], Yb[name] = m.rollout(Th), m.rollout(Tb)
res = pd.DataFrame({n: {**H.horizon_metrics(Yh[n], Th), **H.battle_metrics(Yb[n], Tb)} for n in mixed}).T
display(
    res[
        [
            "health_2s",
            "health_4s",
            "margin_r2_2s",
            "margin_r2_4s",
            "alive_4s",
            "distance_4s",
            "battle_health_nrmse",
            "winner_acc",
            "duration_err_s",
        ]
    ].round(3)
)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
hs = [1, 2, 4, 8]
for ax, metric, title in (
    (axes[0], "health_{}s", "health error (ten marines' hp)"),
    (axes[1], "margin_r2_{}s", "margin R²"),
):
    for n in mixed:
        ax.plot(hs, [res.loc[n, metric.format(h)] for h in hs], "o-", color=COLORS[n], label=n)
    ax.set_xscale("log", base=2)
    ax.set_xticks(hs, [f"{h} s" for h in hs])
    ax.set_title(title)
axes[1].axhline(0, color="k", lw=0.5)
axes[0].legend()
plt.tight_layout()

# %% [markdown]
# **Reading.** Unlike one step ahead (where the per-type state barely helped total health), rollouts gain:
# health −6% (paired 95% [−0.0058, −0.0038]), margin R² at 4 s 0.05 → 0.17-0.23, winner accuracy 0.65 →
# 0.70. The composition changes during a battle (marines die first), and a team-level k cannot follow that.
#
# **Caveat:** alive error is no better than the team model's. The per-type alive law has low exponents
# (0.37-0.53) and low one-step R² (0.05-0.16); it is the weakest part.

# %% [markdown]
# ## 5. What a battle looks like per type
#
# The battle with the median whole-battle error of the typed model; types per side, truth vs. typed (closure).

# %%
Yb_t = mixed["typed (closure)"].rollout(Bb)
end = Tb["end"]
start = np.maximum(Tb["Y"][:, 0, :2], 1e-6)
mask = np.arange(Tb["Y"].shape[1])[None, :] <= end[:, None]
err = np.where(mask[:, :, None], (team_view(Yb_t)[:, :, :2] - Tb["Y"][:, :, :2]) / start[:, None, :], 0)
nr = np.sqrt((err**2).sum((1, 2)) / (2 * (end + 1)))
i = int(np.argmin(np.abs(nr - np.median(nr))))
t = np.arange(end[i] + 1) * DT
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, s in zip(axes[:2], SIDES, strict=False):
    for tt, ls in zip(TYPES, ("-", "--"), strict=True):
        ax.plot(
            t, Bb["Y"][i, : end[i] + 1, idx("health", s, tt)], color="k", ls=ls, lw=2, label=f"{tt} truth"
        )
        ax.plot(t, Yb_t[i, : end[i] + 1, idx("health", s, tt)], color="#d62728", ls=ls, label=f"{tt} typed")
    ax.set_title(f"{s}: health per type (ten full units of the type)")
    ax.legend(fontsize=7)
ax = axes[2]
ax.plot(t, Tb["Y"][i, : end[i] + 1, 0] - Tb["Y"][i, : end[i] + 1, 1], color="k", lw=2, label="truth")
for n in mixed:
    Z = team_view(Yb_t) if n == "typed (closure)" else Yb[n]
    ax.plot(t, Z[i, : end[i] + 1, 0] - Z[i, : end[i] + 1, 1], color=COLORS[n], label=n)
ax.axhline(0, color="k", lw=0.5)
ax.set_title(f"team margin (battle {Bb['ep'][i]}, typed nRMSE {nr[i]:.3f})")
ax.legend(fontsize=7)
plt.tight_layout()

# %% [markdown]
# **Reading (this battle).** Blue's marines are predicted to lose health too slowly and Red's marines too fast,
# so every model calls the wrong winner here although the per-type curves start well. The error is in *which*
# units take the damage, not only how much: the composition is what the per-type state is for, and also where
# it still fails.

# %% [markdown]
# ## 6. Claim: target choice is readable
#
# Measured share of each attacker type's fire on marauders against the marauders' share of the target side's
# alive units (binned, steps with fire), with the fitted closure at front offset 0 and the proportional rule
# (diagonal).

# %%
from scipy.special import expit, logit  # noqa: E402

fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
for ax, (o, a) in zip(axes, [(o, a) for o in SIDES for a in TYPES], strict=True):
    s = "red" if o == "blue" else "blue"
    pi = val8[f"alive_{s}_marauder"] / (val8[f"alive_{s}_marine"] + val8[f"alive_{s}_marauder"])
    fm, fM = val8[f"fire_{o}_{a}_marine"], val8[f"fire_{o}_{a}_marauder"]
    ok = (fm + fM > 0) & (pi > 0) & (pi < 1)
    bins = pd.cut(pi[ok], np.linspace(0, 1, 11))
    meas = fM[ok].groupby(bins, observed=True).sum() / (fm + fM)[ok].groupby(bins, observed=True).sum()
    ax.scatter([b.mid for b in meas.index], meas.values, color="k", s=18, label="measured")
    grid = np.linspace(0.02, 0.98, 50)
    b = typed.alloc_b[(o, a)]
    ax.plot(grid, expit(b[0] + b[1] * logit(grid)), color="#d62728", label="closure (front offset 0)")
    ax.plot(grid, grid, color="#999999", ls="--", label="proportional")
    ax.set_title(f"{a} fire ({o})\nb = {np.round(b, 2).tolist()}", fontsize=9)
    ax.set_xlabel("marauder share of target side")
axes[0].set_ylabel("share of fire on marauders")
axes[0].legend(fontsize=7)
plt.tight_layout()

# %% [markdown]
# **Reading.** b2 (front offset) is 1.7-2.2 for all four: fire shifts to whichever type stands further forward,
# as a nearest-target rule implies. Red's intercept is positive (Red over-targets marauders); Blue's is ≈ 0.
#
# **Caveat:** in rollouts the fitted closure and the proportional rule are about equally good (closure a little
# better per type, the rule a little better on the team margin).

# %% [markdown]
# ## 7. Claim: one distance is defensible on average
#
# Marauder-minus-marine front offset (map units; + = marauders further forward) around first contact, val.

# %%
rows = []
for _ep, g in val8.groupby(level=0):
    rb = g[[c for c in g if c.startswith("reach_")]].sum(axis=1).to_numpy()
    if not (rb > 0).any():
        continue
    k0 = int(np.argmax(rb > 0))
    for s in SIDES:
        gap = (g[f"front_{s}_marauder"] - g[f"front_{s}_marine"]).to_numpy()
        for dk in (-16, 0, 16, 32, 64):
            if 0 <= k0 + dk < len(g):
                rows.append(
                    {
                        "side": s,
                        "s from first contact": dk * DT,
                        "offset": gap[k0 + dk],
                        "depth": g[f"depth_{s}"].iloc[k0 + dk],
                    }
                )
pd.DataFrame(rows).groupby(["side", "s from first contact"]).agg(["mean", "std"]).round(2)

# %% [markdown]
# **Reading.** Both types walk at 3.15, and on average they stay level (|mean| ≤ 0.2). Per battle the offset
# varies about as much as the formation depth, and the model holds it at its start value. Decision (19 Nov):
# defensible on average; no offset state. With faster stalkers this would not hold.

# %% [markdown]
# ## 8. Ranking intents (counterfactual branches, exploratory)

# %%
cf_states, labels = cf_ranking.load_cf()
Bc = cf_ranking.branch_batch(cf_states)
preds = cf_ranking.predictions(Bc, mixed)
for h in (4, 8):
    info, table = cf_ranking.ranking(Bc, preds, labels, h)
    display(
        Markdown(
            f"**At {h} s** — {info['seeds']} seeds, spread of the true margin change "
            f"{info['spread']:.3f}, best fixed intent: {info['best fixed']}"
        )
    )
    display(table.round(3))

# %% [markdown]
# **Reading.** Same pattern as marines: rank correlation ≈ 0.5 at 4 s and ≈ 0.25 at 8 s, well above random,
# but no model picks the best intent more often than "always advance/nearest" (0.41-0.42 vs 0.51), and the
# per-type state does not help ranking (the team model ranks slightly better).

# %% [markdown]
# ## Summary and open questions
#
# | claim | supported? | strength |
# |---|---|---|
# | Damage matrix matches the game's pattern | yes, all 8 entries, tight bootstrap | needs measured target choice to fit; per-firing-unit rate not identifiable with reach in the library |
# | Firing share depends on type and battle | marines 0.79 alone / 0.82-0.86 mixed; marauders 0.53-0.54 alone / 0.58-0.60 mixed | cause unknown |
# | Structure transfers to marauders | same 11 terms on refit | scaling the numbers fails (firing share) |
# | Per-type state helps rollouts | health −6%, margin R² 4 s +0.12-0.18, winner +0.05 | alive law weak; only val |
# | Target choice readable | front offset and Red's bias | closure ≈ proportional rule in rollouts |
# | One distance | level on average | per-battle spread |
# | Intent ranking | above random | not above the best fixed intent; per-type state doesn't help |
#
# Open: why marauders in reach fire less; a better per-type alive law; whether the firing share should be
# an explicit part of the model; gates and a pre-registered test3 run if this becomes a thesis chapter.
