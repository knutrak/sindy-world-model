# ruff: noqa: E402, E501, E731, B018, B023, B905, I001
# %% [markdown]
# # RQ1. World model identification
#
# > *Can SINDy identify a compact and interpretable world model of small-scale combat dynamics directly from game data?*
#
# This notebook collects the evidence. It states what was measured and how; **the answer is left for you** (last cell).
#
# | section | question | data | status |
# |---|---|---|---|
# | 1 | what are the equations? | frozen fits (train) | held-out tested, except stalkers and triple |
# | 2 | how much did SINDy choose, and how much was designed? | models | computed here |
# | 3 | do the coefficients match the game? | held-out refits (recorded) and triple test refit | pre-registered |
# | 4 | is the structure stable? | train, bootstrap by battle | exploratory, 25 Nov |
# | 5 | what does compactness cost? | marine cross-validation (recorded) | exploratory |

# %%
import sys

sys.path.insert(0, "../lib")
import rq
from rq import AQUA, BLUE, GREY, INK, ORANGE, np, pd, plt
from IPython.display import Markdown, display

from frozen_models import equations

M = rq.models()

# %% [markdown]
# ## 1. The equations
# Health unit: ten full units of the type = 1. `R[a]` enemy units in reach (tens), `s[t]` share of fire on type t. Coefficients
# are full units of the target removed per second per attacker in reach.

# %%
for name, m in M.items():
    display(Markdown(f"**{name}**\n```\n{equations(m)}\n```"))

# %% [markdown]
# ## 2. Chosen by SINDy, or designed?
# *Candidates* are offered to STLSQ per health equation; *kept* is what it selected. *Fitted outside SINDy* counts the closure
# numbers that actually change a rollout (each is perturbed in turn). `Pipeline.complexity()` reports other closure counts
# (last column): it omits the gates and the withdraw closure and counts unused target-choice numbers.

# %%
rows = []
for name, m in M.items():
    n_eq = len(m.coef)
    cand = len(next(iter(m.coef.values())))
    kept = sum(bool(v) for c in m.coef.values() for v in c.values())
    used = rq.used_closure_numbers(m)
    rows.append(
        {
            "model": name,
            "health equations": n_eq,
            "candidates each": cand,
            "health terms kept (all)": kept,
            "distance terms kept": int((m.dist_coef != 0).sum()),
            "alive numbers (least squares)": sum(1 + (f != 0) for _, f in m.p.values()),
            **{f"{k} (outside)": v for k, v in used.items()},
            "outside, total": sum(used.values()),
            "complexity() says": m.complexity()["closure numbers"],
        }
    )
design = pd.DataFrame(rows).set_index("model")
design

# %% [markdown]
# Not fitted at all, but given: the definition of reach (attack range + radii, capped at 6), withdraw as rules, target
# choice proportional to type shares, the form of every closure and of the alive law.

# %% [markdown]
# ## 3. Coefficients against the game
# Firepower and walking speeds of models **refitted on each held-out set** (recorded, pre-registered bands shaded), and the triple
# model refitted on its own held-out set. Damage, cooldown, armor and speed are never given to the pipeline.

# %%
T = rq.held_out()
pts = []
for s in ("marines", "marauders", "mixed", "uneven"):
    for k, v in T[s]["refit_physics"]["fitted / game"].items():
        if k != "distance close*wd":
            pts.append(
                {
                    "set": s,
                    "term": k.replace("distance ", ""),
                    "ratio": v,
                    "kind": "speed" if k.startswith("distance") else "firepower",
                }
            )
txt = (rq.DOCS / "triple" / "02_test.md").read_text().split("## Firepower")[1].split("```")[1]
for line in txt.splitlines():
    p = line.rsplit(None, 3)
    if len(p) == 4 and "→" in p[0]:
        pts.append({"set": "triple", "term": p[0].strip(), "ratio": float(p[3]), "kind": "firepower"})
P = pd.DataFrame(pts)
order = ["marines", "marauders", "mixed", "uneven", "triple"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, kind, band in ((axes[0], "firepower", (0.75, 1.0)), (axes[1], "speed", (0.8, 1.2))):
    ax.axvspan(*band, color=AQUA, alpha=0.10, lw=0)
    ax.axvline(1, color=INK, lw=0.8)
    for j, s in enumerate(order):
        r = P[(P.set == s) & (P.kind == kind)]["ratio"].to_numpy()
        ax.scatter(
            r,
            j + np.linspace(-0.22, 0.22, len(r)) if len(r) > 1 else [j] * len(r),
            s=34,
            color=BLUE,
            edgecolor="white",
            lw=0.8,
        )
    ax.set_yticks(range(len(order)), order)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("fitted / game value")
    ax.set_title(f"{kind} (band fixed before the test)")
axes[0].invert_yaxis()
plt.tight_layout()
plt.show()
P.groupby(["kind", "set"]).ratio.agg(["min", "max", "count"]).round(3)

# %% [markdown]
# Speeds are not shown for triple (two walking speeds, reported not judged) and for stalkers alone, whose advance coefficient is
# 3.97 against 8.26 in the game (`docs/triple/04_stress.md`).

# %% [markdown]
# ## 4. Is the structure stable?
# Health equations refitted on 60 battle-bootstrap resamples of train, at several thresholds. "Exact" = exactly the firepower terms,
# nothing else, in every equation. The reported models use 0.15.

# %%
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import threshold as TH

thr = (0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.25)
res = {}
gap = []
for name in TH.NAMES:
    m, nb, F = TH.grams(name)
    rng = np.random.default_rng(0)
    W = [np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(60)]
    S = {k: [TH.scaled(np.tensordot(w, G, 1), names) for w in W] for k, (names, G) in F.items()}
    ex = []
    for t in thr:
        ok = 0
        for j in range(60):
            ok += all(
                {names[i] for i in np.flatnonzero(TH.select(*S[k][j], t))}
                == {f"R[{a}]*s[{k[1]}]" for a in m.L.types}
                for k, (names, _) in F.items()
            )
        ex.append(ok / 60)
    res[name] = ex
    real, spur = [], []
    for k, (names, G) in F.items():
        A, b, sd, n = TH.scaled(G.sum(0), names)
        want = np.array([x in {f"R[{a}]*s[{k[1]}]" for a in m.L.types} for x in names])
        i = np.flatnonzero(want)
        real += list(np.abs(np.linalg.solve(A[np.ix_(i, i)], b[i])) / sd[len(names)])
        full = np.linalg.solve(A + 1e-6 * n * np.eye(len(names)), b)
        spur += list(np.abs(full[~want]) / sd[len(names)])
    gap.append(
        {
            "model": name,
            "smallest real effect": min(real),
            "largest spurious effect": max(spur),
            "exact at 0.15": ex[thr.index(0.15)],
        }
    )
E = np.array(list(res.values()))
fig, ax = plt.subplots(figsize=(8.5, 2.8))
ax.imshow(E, cmap="Blues", vmin=0, vmax=1.6, aspect="auto")
ax.grid(False)
for i in range(E.shape[0]):
    for j in range(E.shape[1]):
        ax.text(j, i, f"{E[i, j]:.0%}", ha="center", va="center", fontsize=9, color=INK)
ax.set_xticks(range(len(thr)), [f"{t:.2f}" + ("\n(used)" if t == 0.15 else "") for t in thr])
ax.set_yticks(range(len(res)), list(res))
ax.set_xlabel("STLSQ threshold (effect size)")
ax.set_title("share of 60 resamples with exactly the right structure")
plt.tight_layout()
plt.show()
pd.DataFrame(gap).set_index("model").round(3)

# %% [markdown]
# ## 5. What does compactness cost?
# Marine models, 5-fold cross-validation (recorded on 18 Nov, `docs/final_exploration/results.csv`): health error at 4 s against the number
# of terms. Terms exclude closure numbers; the MLPs are drawn at their number of weights.

# %%
r = pd.read_csv(rq.DOCS / "final_exploration" / "results.csv")
cv = (
    r[r.split == "cv"]
    .groupby("experiment")
    .agg(health=("health_4s", "mean"), terms=("n_terms", "first"), nrmse=("battle_health_nrmse", "mean"))
)
fig, ax = plt.subplots(figsize=(8, 3.8))
kind = lambda e: (
    "MLP"
    if e.startswith("J/")
    else "baseline"
    if "lanchester" in e or "persistence" in e or "no-control" in e
    else "SINDy variants"
)
for k, c in (("SINDy variants", BLUE), ("baseline", GREY), ("MLP", ORANGE)):
    d = cv[[kind(e) == k for e in cv.index]]
    ax.scatter(np.maximum(d.terms, 0.7), d.health, s=40, color=c, label=k, edgecolor="white", lw=0.8)
for e, dy in (
    ("ref/A", 0.006),
    ("ref/S", -0.009),
    ("J/mlp-wide", 0.006),
    ("ref/lanchester-square", 0.006),
    ("ref/persistence", -0.012),
    ("ref/no-control", 0.006),
):
    ax.annotate(e.split("/")[1], (max(cv.terms[e], 0.7), cv.health[e] + dy), fontsize=8, ha="center")
ax.set_xscale("log")
ax.set_xlabel("terms (log scale)")
ax.set_ylabel("health error at 4 s")
ax.legend()
plt.tight_layout()
plt.show()
cv.loc[
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
].round(4)

# %% [markdown]
# Recorded elsewhere (val, `docs/2026-11-10_sindyc-intents-exploration.md`): a generic degree-2 SINDYc kept 138 terms and reached 0.053 at 4 s,
# against 0.057 for the sparse physics model of that date. Same accuracy, not readable.

# %% [markdown]
# ## Caveats
# - The library is designed from the game's rules, so "SINDy found" means *selected from a designed set*.
# - Until 25 Nov a defect in `stlsq` made the constant impossible to keep. With it fixed all terms and numbers are unchanged, but the
#   constant is close to the threshold in the single-type models (`docs/triple/04_stress.md`).
# - True terms with a small effect size are not found (a firepower of 0.03: 0 of 20 trials).
# - On 17 Nov the pre-registered gates rejected model A and passed B; A\*, A\*-2k and the pipeline came later.
# - The triple expectations were Claude's and were collected with uncommitted code.

# %% [markdown]
# ## Your answer (to write)
# 1. In one sentence: what did SINDy *discover*, as opposed to what you designed?
# 2. Is 10 terms plus 23 outside numbers "compact" for marines? Compact compared with what?
# 3. Is a term part of the model because it matters for prediction, or because it is really there?
# 4. Which result here would have made you answer "no"?
#
# *Answer:*
