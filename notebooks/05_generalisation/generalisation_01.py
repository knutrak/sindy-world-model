# ruff: noqa: E501, B018  (markdown tables and display-only cell endings)
# %% [markdown]
# # One equation-discovery pipeline across unit types
#
# **Claim.** The same pipeline, given only a scenario's battles and its units' game statistics, finds equations
# with the same structure, coefficients that match the game's rules, and comparable predictive power: for
# marines, marauders and mixed armies, and for an uneven-mix scenario it had never seen.
#
# **Evidence.** A pre-registered test, run once on four held-out sets (`docs/generalisation/03_preregistration.md`,
# `04_test_results.md`): **19 of 20 expectations met**. Report: `docs/generalisation/REPORT.md`.
#
# | # | claim | evidence here | caveat |
# |---|---|---|---|
# | 1 | One pipeline, the same structure everywhere | the frozen equations, side by side | the target choice and withdraw are rules, not fitted |
# | 2 | Coefficients match the game | firepower and speeds ÷ game values, four held-out refits | effective rates (× share of units firing); one term at 1.02 |
# | 3 | It predicts, within 2-3 s | held-out error vs horizon against baselines | 2-3 s horizon; no black-box comparison here |
# | 4 | Per-type states pay off in mixed armies | per-type vs team-level, paired intervals | the per-type alive law is weak |
# | 5 | Held-out matches val | val vs test, health 2 s | — |
# | 6 | How the test was kept honest | the timeline of commits | one amendment before the read |

# %%
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from sindy_wm.evaluation import harness as H
from sindy_wm.paths import PROJECT_ROOT

DOCS = PROJECT_ROOT / "docs" / "generalisation"
test = json.loads((DOCS / "04_test_results.json").read_text())
val = json.loads((DOCS / "02_val_table.json").read_text())
models = pickle.load(
    open(H.CACHE / "generalisation_frozen.pkl", "rb")
)  # experiments/generalisation/frozen_models.py
SCEN = ["marines", "marauders", "mixed", "uneven"]
COL = {"marines": "#1f77b4", "marauders": "#ff7f0e", "mixed": "#2ca02c", "uneven": "#d62728"}

# %% [markdown]
# ## 1. One pipeline, the same structure everywhere
#
# The frozen models, each fitted on its own scenario's train set (`configs/generalisation_pipeline.yaml`). Units:
# target units removed per second per enemy unit in reach; R terms of Blue attackers count R·(1 − wd); s[t] is the
# share of fire on type t (proportional to the target side's type shares).

# %%
side = {"blue": "B", "red": "R"}
for name, m in models.items():
    lines = []
    for (s, t), c in m.coef.items():
        lines.append(f"H_{side[s]}^{t}' = " + "  ".join(f"{v:+.3f}·{k}" for k, v in c.items() if v))
    for (s, t), (p, pf) in m.p.items():
        lines.append(
            f"(ln A_{side[s]}^{t})' = ({p:.3f}{f' {pf:+.3f}·focus' if pf else ''})·(ln H_{side[s]}^{t})'"
        )
    lines.append(
        "d' = " + "  ".join(f"{c:+.3f}·{n}" for n, c in zip(m.dist_names, m.dist_coef, strict=True) if c)
    )
    display(Markdown(f"**{name}** ({m.complexity()['terms']} terms)\n```\n" + "\n".join(lines) + "\n```"))

# %% [markdown]
# **Reading.** The same building blocks in every scenario: one damage term per attacker type and target type,
# an alive exponent per type, and the same five distance terms. Only the numbers differ. The uneven scenario has
# no model of its own: the mixed model is applied unchanged.

# %% [markdown]
# ## 2. Claim: the coefficients match the game
#
# The pipeline refitted on each *held-out* set on its own (what it discovers there), divided by the game's
# values. Firepower: (damage + bonus − armor) / cooldown (rounded up to ticks) / target hp. Speeds: walking
# speed 3.15 (one side walks) or 6.3 (both walk). Pre-registered bands: firepower 0.75-1.0, speeds 0.8-1.2.

# %%
rows = []
for s in SCEN:
    ph = test[s]["refit_physics"]["fitted / game"]
    for k, v in ph.items():
        if k == "distance close*wd":
            continue  # not a term with the withdraw rule
        rows.append(
            {
                "scenario": s,
                "term": k,
                "ratio": v,
                "kind": "speed" if k.startswith("distance") else "firepower",
            }
        )
R = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2, 1]})
for ax, kind, band in ((axes[0], "firepower", (0.75, 1.0)), (axes[1], "speed", (0.8, 1.2))):
    sub = R[R.kind == kind]
    terms = list(dict.fromkeys(sub.term))
    for j, s in enumerate(SCEN):
        d = sub[sub.scenario == s].set_index("term").reindex(terms)
        ax.scatter(d["ratio"], np.arange(len(terms)) + (j - 1.5) * 0.15, color=COL[s], label=s, s=28)
    ax.axvspan(*band, color="#2ca02c", alpha=0.08)
    ax.axvline(1, color="k", lw=0.6)
    ax.set_yticks(range(len(terms)), [t.replace("distance ", "") for t in terms], fontsize=8)
    ax.set_xlabel("fitted / game")
    ax.set_title(f"{kind} (band: pre-registered)")
axes[0].legend(fontsize=8)
plt.tight_layout()

# %% [markdown]
# **Reading.** Every firepower term lies at 0.77-0.94 of the game rate except one: marine → marauder in the
# uneven scenario at 1.02 (the one missed expectation; cause unknown). Below 1 is expected: these are rates per
# unit *in reach*, and not every unit in reach fires all the time. Speeds lie at 0.81-1.04.
#
# **Caveat:** reach is counted at SMAClite's real firing limit of 6. Counted at the weapon range, marauders came
# out at 57-64% of the game rate (`docs/mixed/PROGRESS.md`).

# %% [markdown]
# ## 3. Claim: it predicts, within 2-3 s
#
# The train-fitted models on the held-out sets, against persistence (no change), Lanchester square, and (mixed,
# uneven) the team-level A\*-2k. Health in ten units of the scenario's first type.

# %%
hs = [1, 2, 4, 8]
fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
style = {
    "pipeline": ("-", 2.0),
    "persistence": (":", 1.2),
    "Lanchester": ("--", 1.2),
    "team-level A*-2k": ("-.", 1.2),
}
for ax, s in zip(axes, SCEN, strict=True):
    t = pd.DataFrame(test[s]["table"])
    for model, (ls, lw) in style.items():
        if model in t.index:
            ax.plot(
                hs,
                [t.loc[model, f"health_{h}s"] for h in hs],
                ls,
                lw=lw,
                marker="o",
                ms=3,
                color=COL[s] if model == "pipeline" else "#555555",
                label=model,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(hs, [f"{h} s" for h in hs])
    ax.set_title(f"{s}: health error (held-out)")
axes[2].legend(fontsize=7)
plt.tight_layout()

summary = pd.DataFrame(
    {
        s: {
            "health 2 s": pd.DataFrame(test[s]["table"]).loc["pipeline", "health_2s"],
            "margin R² 2 s": pd.DataFrame(test[s]["table"]).loc["pipeline", "margin_r2_2s"],
            "battle nRMSE": pd.DataFrame(test[s]["table"]).loc["pipeline", "battle_health_nrmse"],
            "reliable horizon (s)": test[s]["horizon"],
        }
        for s in SCEN
    }
).T
summary.round(4)

# %% [markdown]
# **Reading.** At 2 s, the error is about a quarter of the "no change" baseline and roughly half of
# Lanchester's, in every scenario, including the never-seen uneven one. The reliable horizon (margin R² ≥ 0.3,
# interval above 0) is 2-3 s, which meets the 1-2 s use case.
#
# **Caveat:** the interpretable model is not a long-range predictor. A black-box MLP was clearly better on
# marines (`docs/final_exploration/REPORT.md`); it is not part of this test.

# %% [markdown]
# ## 4. Claim: per-type states pay off in mixed armies

# %%
ci = {s: test[s]["ci_model_minus_baseline"] for s in ("mixed", "uneven")}
pd.DataFrame(
    {
        s: {k: f"[{v[0]:+.4f}, {v[1]:+.4f}]" for k, v in d.items() if k.startswith("team")}
        for s, d in ci.items()
    }
)

# %% [markdown]
# Paired 95% intervals of (per-type − team-level) health error: below 0 in both mixed scenarios at 2 and 4 s.
# On mixed test3 the margin R² at 4 s is 0.19 against 0.08. The per-type state can follow the composition as it
# changes (marines die first); a team-level k cannot.
#
# **Caveat:** the per-type alive law is weak (exponents 0.38-0.58), and the alive error is about the same as the
# team-level model's.

# %% [markdown]
# ## 5. Claim: held-out matches val

# %%
val_ref = {"marines": 0.0277, "marauders": 0.0203, "mixed": 0.0346}
cmp = pd.DataFrame(
    {
        s: {"val": v, "held-out": pd.DataFrame(test[s]["table"]).loc["pipeline", "health_2s"]}
        for s, v in val_ref.items()
    }
).T
cmp["held-out / val"] = cmp["held-out"] / cmp["val"]
cmp.round(4)

# %% [markdown]
# Within 2% in all three scenarios: no sign that the choices made on val overfitted to it.

# %% [markdown]
# ## 6. How the test was kept honest
#
# | when | what | commit |
# |---|---|---|
# | before WP4 | decision rules for withdraw and target choice | `c2a8c9e` |
# | after WP4-5 | config frozen | `5da772e` |
# | before any held-out read | pre-registration and test script | `c6b8e2c` |
# | after a val dry run | amendment: withdraw movement by rule (the fitted gate failed on small sets) | `c135280` |
# | after a second dry run | script bug fixed (checked a term that no longer exists) | `7030381` |
# | once | held-out test: 19 of 20 met | `21348eb` |

# %%
exp_rows = []
for s in SCEN:
    for k, v in test[s]["expectations"].items():
        exp_rows.append({"scenario": s, "expectation": k, "met": v})
pd.DataFrame(exp_rows).pivot_table(index="expectation", columns="scenario", values="met", aggfunc="first")[
    SCEN
]

# %% [markdown]
# ## Summary
#
# | claim | supported? | caveat |
# |---|---|---|
# | Same structure everywhere | yes, in all four held-out refits | target choice and withdraw are rules |
# | Coefficients match the game | 0.77-0.94 (one 1.02) firepower, 0.81-1.04 speeds | effective rates; reach at SMAClite's limit 6 |
# | Predicts within 2-3 s | beats persistence and Lanchester everywhere | not a long-range model |
# | Per-type helps mixed armies | yes, at 2 and 4 s | per-type alive law weak |
# | Held-out ≈ val | within 2% | one amendment before the read |
