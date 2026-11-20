# ruff: noqa: E501, B018  (markdown tables and display-only cell endings)
# %% [markdown]
# # Marines and marauders: what changes when a battle has two unit types
#
# Branch `mixed-marauders` (19-20 Nov). The marine model (A\*-2k, `notebooks/03_final_exploration/final_01_model_showcase`) is the
# starting point. Two data sets: **exp09** (marauders only) and **exp08** (marines + marauders, the same mix
# on both sides, marauder share ¼, ⅓ or ½). Exploratory, on `val` and `cf`; the held-out sets (exp09 `test`,
# exp08 `test3`) have not been read. Log: `docs/mixed/PROGRESS.md`.
#
# **Reach is counted at min(attack reach, 6)** (updated 20 Nov). SMAClite only offers agents attack actions
# within 6, whatever the unit's range, so a marauder (attack reach 7.125) can't be ordered to fire between 6
# and 7.125 (section 2). Every model here uses the capped reach, and the per-type model adds a **withdraw
# rule** taken from the scripts, with nothing fitted: Blue units don't fire while Blue withdraws.
#
# | # | claim | evidence | caveat |
# |---|---|---|---|
# | 1 | The per-type damage matrix matches the game | all 8 entries 0.82-0.94 of the game rate, Blue and Red alike | needs measured target choice to fit |
# | 2 | Counted at reach 6, the firing share is the same for both types | marauders 0.54 → ≈ 0.85 | the cap is an environment limit, not physics |
# | 3 | The marine model transfers to marauders | scaled without refitting, margin R² 2 s 0.36 | the capped refit predicts a bit worse than the uncapped one |
# | 4 | The per-type state pays off in rollouts | beats A\*-2k on team totals | only with the withdraw rule; alive law weak |
# | 5 | What a battle looks like per type | one median battle | — |
# | 6 | Target choice is readable | fire shifts to the type further forward; Red over-targets marauders | fitted closure ≈ proportional rule |
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
P8 = DATA_DIR / "processed" / "exp08_mixed_r6"
cf_ranking.P = P8
mixed = pickle.load(open(H.CACHE / "mixed_models_r6.pkl", "rb"))  # all fitted on exp08_mixed_r6 train
MAIN = "typed (closure) + wd rule"
typed = mixed[MAIN]
COLORS = {
    "team A*-2k": "#7f7f7f",
    "typed (closure)": "#f4a582",
    "typed (rule)": "#fddbc7",
    "typed (closure) + wd rule": "#d62728",
    "typed (rule) + wd rule": "#b2182b",
}

# %% [markdown]
# ## 0. What changed from the marine model
#
# | | marine A\*-2k | mixed per-type model (`TypedA`) |
# |---|---|---|
# | state | H_B, H_R, A_B, A_R, d | H and A **per type** and side, plus d (9) |
# | health | H_B' = k_B·R_R | **2 × 2 matrix per side**: Σ_a k(a→t)·R_a·σ(a→t) |
# | target choice | — | **closure σ**: share of type a's fire on marauders, logistic in the target side's marauder share and front offset |
# | reach | one closure per side, ρ = 5.75 | **per type**, ρ = the type's reach capped at 6 (≈ 5.84 marines, 6.0 marauders), shifted by how far forward the type sits |
# | withdraw | fitted k_wd on Blue's fire | **rule**: Blue units don't fire while withdrawing (the scripts' behaviour; nothing fitted) |
# | alive | p per side | p per type and side |
# | distance | marine gates | the same equation, refitted on mixed data |
# | unchanged | | one-step STLSQ, A\*-2k's alive-law form, offset-decaying reach, team-wide intents, shape held |
#
# The groups are unit *types*, not spatial clusters: units are shuffled within the formation and share one
# centroid distance.

# %% [markdown]
# ## 1. Claim: the damage matrix matches the game
#
# Fitted (one step, measured reach × measured share of fire) against the game's rate per *firing* unit:
# (damage + bonus − armor) / cooldown rounded up to ticks / target hp. Units: target units removed per second.
# The fitted value is per unit *in reach*, so it is the game rate times the share of units in reach that fire.

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
m_ = matrix.reset_index()
lab = [f"{r.attacker.split()[0][:3]}→{r.target[:3]}\n{r.attacker.split()[1]}" for r in m_.itertuples()]
x = np.arange(len(matrix))
ax.bar(x - 0.2, matrix["game"], 0.4, label="game, per firing unit", color="#bbbbbb")
ax.bar(x + 0.2, matrix["fitted"], 0.4, label="fitted, per unit in reach (≤ 6)", color="#d62728")
ax.set_xticks(x, lab, fontsize=8)
ax.set_ylabel("target units / s")
ax.legend()
plt.tight_layout()

# %% [markdown]
# **Reading.** Every entry is 0.82-0.94 of the game rate, and Blue and Red agree (e.g. marauder → marine
# 0.163 / 0.165). The pattern is the game's: marines barely hurt marauders (0.060 against 0.064), and the
# marauder's +10 against armored shows in marauder → marauder. With reach 7.125 (19 Nov) the marauder
# entries were only 0.60-0.71 of the game rate and Red looked weaker than Blue.
#
# **Caveat:** the rate *per firing unit* is not identifiable when reach is also in the library (firing ≈
# 0.85 × reach); which of two collinear descriptions STLSQ keeps depends on the library.

# %% [markdown]
# ## 2. Claim: counted at reach 6, the firing share is the same for both types
#
# Share of units in reach that are firing, with reach counted at the attack reach and at min(attack reach,
# 6). Single-type battles: 200 val battles each; mixed: the val tables.

# %%
share_rows = []
for label, exp, t in (
    ("marines alone", "exp06_intents", "MARINE"),
    ("marauders alone", "exp09_marauders", "MARAUDER"),
):
    for cap in (None, 6.0):
        s = load_typed_states(RAW_DIR / exp / "val", (t,), episode_ids=list(range(200)), max_reach=cap)
        tl = t.lower()
        for side in SIDES:
            ok = s[f"reach_{side}_{tl}"] > 0
            share_rows.append(
                {
                    "battles": label,
                    "side": side,
                    "type": tl,
                    "reach": "≤ 6" if cap else "attack reach",
                    "firing / in reach": s.loc[ok, f"fire_{side}_{tl}_{tl}"].sum()
                    / s.loc[ok, f"reach_{side}_{tl}"].sum(),
                }
            )
for cap, folder in ((None, "exp08_mixed"), (6.0, "exp08_mixed_r6")):
    v = pd.read_parquet(DATA_DIR / "processed" / folder / "val.parquet")
    for side in SIDES:
        for a in TYPES:
            ok = v[f"reach_{side}_{a}"] > 0
            fire = v[f"fire_{side}_{a}_marine"] + v[f"fire_{side}_{a}_marauder"]
            share_rows.append(
                {
                    "battles": "mixed",
                    "side": side,
                    "type": a,
                    "reach": "≤ 6" if cap else "attack reach",
                    "firing / in reach": fire[ok].sum() / v.loc[ok, f"reach_{side}_{a}"].sum(),
                }
            )
pd.DataFrame(share_rows).pivot_table(
    index=["battles", "type", "side"], columns="reach", values="firing / in reach"
).round(3)

# %% [markdown]
# **Reading.** Counted at the attack reach, marauders fire only about half the time they are in reach. The
# cause is in SMAClite's code: agents only get attack actions against enemies within `AGENT_TARGET_RANGE = 6`
# (one constant for every unit type; SMAC uses a fixed shoot range of 6 too), and Red's AI only picks up new
# targets within its scan radius (5 + target radius). Between 6 and 7.125, Blue's marauders fire 0% of the time
# and Red's 7% (`experiments/mixed/firing_share.py`). Counted at 6, the marauders' share rises to about the
# marines'. For marines the cap never binds (5.75 < 6).

# %% [markdown]
# ## 3. Claim: the marine model transfers to marauders (with the reach capped)
#
# Marauders alone (exp09 `val`, 500 battles): the marine A\*-2k scaled without refitting
# (`switch_library.scaled`: reach, walking speed, time to kill) and A\*-2k's recipe refitted on exp09 `train`,
# each with reach 7.125 and with reach 6.

# %%
marine = pickle.load(open(H.CACHE / "final_Astar2k.pkl", "rb"))


def roll9(m, B):
    return m.simulate(B["X0"], B["shape0"], B["U"], reach0=B["reach0"])


r9, coefs9 = {}, {}
for cap, exp, unit, refit_file in (
    ("7.125", "exp09_marauders", "marauder", "marauders_Astar2k.pkl"),
    ("6", "exp09_marauders_r6", "marauder_r6", "marauders_Astar2k_r6.pkl"),
):
    v9 = H.load_states("val", exp)
    Bh9, Bb9 = H.horizon_batch(v9), H.battle_batch(v9)
    refit = pickle.load(open(H.CACHE / refit_file, "rb"))
    for name, m in (
        (f"marine scaled, reach {cap}", scaled(marine.m, "marine", unit)),
        (f"refit, reach {cap}", refit.m),
    ):
        r9[name] = {**H.horizon_metrics(roll9(m, Bh9), Bh9), **H.battle_metrics(roll9(m, Bb9), Bb9)}
        coefs9[name] = m
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


pd.DataFrame({"marine A*-2k": terms(marine.m)} | {n: terms(m) for n, m in coefs9.items()}).round(3)

# %% [markdown]
# **Reading.** With reach 6 the refit's firepower is 0.125 / 0.122, 90-92% of the game's 0.135 (reach 7.125:
# 57-64%), and the scaled marine model transfers (margin R² 2 s 0.36 against −0.04). The structure is the same
# in every fit (the same terms, apart from the withdraw term).
#
# **Caveat:** the capped refit predicts a little worse than the uncapped one (health 4 s 0.041 vs 0.034). The
# uncapped fit absorbs the unusable range into k, and keeps a withdraw term the capped one drops.

# %% [markdown]
# ## 4. Claim: the per-type state pays off in rollouts (with the withdraw rule)
#
# Mixed `val` (1000 battles), reach 6 throughout. Team-level metrics in ten marines' hp, comparable with the
# marine-only numbers.

# %%
val8 = pd.read_parquet(P8 / "val.parquet")
val8c = add_controls(val8)
Bh = typed_batch(val8c)
Th = team_horizon_batch(Bh)
Bb, Tb = team_battle_batch(val8c)
Yh, Yb = {}, {}
for name, m in mixed.items():
    if name.startswith("typed"):
        Yh[name], Yb[name] = team_view(m.rollout(Bh)), team_view(m.rollout(Bb))
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

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
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
axes[0].legend(fontsize=7)
plt.tight_layout()

# %% [markdown]
# **Reading.** With the withdraw rule the per-type model beats the team model: health 4 s 0.065 against
# 0.072 (paired 95% [−0.008, −0.005]), margin R² 4 s 0.10-0.15 against 0.04. Without the rule it loses that
# advantage: under withdraw it predicts Blue dealing 2.5 times the true damage, because the reach closure only
# sees distance and the capped fit keeps no withdraw term.
#
# **Caveats:** the reach-7.125 fits (19 Nov) predicted a little better still (margin R² 4 s 0.17-0.23, battle
# nRMSE 0.125-0.127). Alive error is no better than the team model's: the per-type alive law is the weakest part.

# %% [markdown]
# ## 5. What a battle looks like per type
#
# The battle with the median whole-battle error of the main model (closure + withdraw rule).

# %%
Yb_t = typed.rollout(Bb)
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
        ax.plot(t, Yb_t[i, : end[i] + 1, idx("health", s, tt)], color="#d62728", ls=ls, label=f"{tt} model")
    ax.set_title(f"{s}: health per type (ten full units of the type)")
    ax.legend(fontsize=7)
ax = axes[2]
ax.plot(t, Tb["Y"][i, : end[i] + 1, 0] - Tb["Y"][i, : end[i] + 1, 1], color="k", lw=2, label="truth")
for n in ("team A*-2k", MAIN, "typed (rule) + wd rule"):
    ax.plot(t, Yb[n][i, : end[i] + 1, 0] - Yb[n][i, : end[i] + 1, 1], color=COLORS[n], label=n)
ax.axhline(0, color="k", lw=0.5)
ax.set_title(f"team margin (battle {Bb['ep'][i]}, nRMSE {nr[i]:.3f})")
ax.legend(fontsize=7)
plt.tight_layout()
u = Bb["U"][i, : end[i] + 1]
display(
    Markdown(
        f"Intents in this battle: advance {u[:, 0].mean():.0%}, hold {u[:, 1].mean():.0%}, "
        f"withdraw {u[:, 2].mean():.0%}, focus {u[:, 3].mean():.0%}."
    )
)

# %% [markdown]
# ## 6. Claim: target choice is readable
#
# Measured share of each attacker type's fire on marauders against the marauders' share of the target side's
# alive units (binned, steps with fire), with the fitted closure at front offset 0 and the proportional rule.

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
# The allocation uses the recorded targets, so it doesn't depend on the reach cap.
#
# **Caveat:** in rollouts the fitted closure and the proportional rule are about equally good.

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
# defensible on average; no offset state.

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
# **Reading.** With the withdraw rule the per-type model ranks like the team model (≈ 0.51 at 4 s, ≈ 0.29 at
# 8 s); without it ranking drops to 0.29 at 4 s, because the withdraw branches are mispredicted. No model picks
# the best intent more often than "always advance/nearest".

# %% [markdown]
# ## Summary and open questions
#
# | claim | supported? | strength |
# |---|---|---|
# | Damage matrix matches the game | all 8 entries 0.82-0.94 of the game rate, Blue and Red alike | needs measured target choice to fit |
# | Firing share is the same for both types at reach 6 | marauders 0.54 → ≈ 0.85 | the cap comes from the environment (AGENT_TARGET_RANGE) |
# | Marine model transfers to marauders | scaled, margin R² 2 s 0.36 | capped refit a little worse than the uncapped one |
# | Per-type state helps rollouts | health −10%, margin R² 4 s +0.06-0.11 vs the team model | needs the withdraw rule; alive law weak; only val |
# | Target choice readable | front offset and Red's bias | closure ≈ proportional rule in rollouts |
# | One distance | level on average | per-battle spread |
# | Intent ranking | above random | not above the best fixed intent |
#
# Open: a better per-type alive law; whether withdraw belongs in the model at all; gates and a pre-registered
# test3 run if this becomes a thesis chapter.
