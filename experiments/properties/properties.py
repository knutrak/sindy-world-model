"""Unit properties, one at a time (docs/properties/00_plan.md, approved 25 Nov 2026).

uv run --extra explore experiments/properties/properties.py g0        V0 plays exactly as the built-in marine
uv run --extra explore experiments/properties/properties.py build     state tables, one per variant
uv run --extra explore experiments/properties/properties.py analyse   once -> docs/properties/01_results.md
"""

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts", "experiments/mixed", "experiments/generalisation", "experiments/triple"):
    sys.path.insert(0, str(ROOT / p))
import evaluate_final as E  # noqa: E402
from build_typed_states import build_one  # noqa: E402
from collect_properties import MAX_SECONDS, STEP_MUL, VARIANTS  # noqa: E402
from common import md  # noqa: E402
from triple import config, equations  # noqa: E402

from sindy_wm.envs.episode import play_episode  # noqa: E402
from sindy_wm.envs.episode_sampling import CloseBattleSampler  # noqa: E402
from sindy_wm.envs.intents import IntentSchedule  # noqa: E402
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds  # noqa: E402
from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.models.intent_model import add_controls  # noqa: E402
from sindy_wm.models.pipeline import SIDES, Layout, Pipeline, prepare  # noqa: E402
from sindy_wm.paths import DATA_DIR  # noqa: E402
from sindy_wm.states.typed import pair_reach, unit_stats  # noqa: E402

PROC, DOCS = DATA_DIR / "processed", ROOT / "docs" / "properties"
OUT = PROC / "exp13_properties_typed"
G0_SEED, SPLIT = 31_900_000, 500  # battles 0-499 refit, 500-999 score
FIREPOWER, SPEED, RANGE = ("DM", "DP", "CM", "CP", "HM", "HP", "A1", "A3"), ("SM", "SP"), ("R4", "R3")
DIST_P3 = ("close*adv", "close*hold*far")


def rate(name: str) -> float:
    """Full units removed per second by one firing unit of this type, against its own type (no shields)."""
    s = unit_stats(name)
    return 1 / (s["hp"] / (s["damage"] - s["armor"]) * np.ceil(s["cooldown"] * 16) / 16)


def rescaled(frozen: Pipeline, v: str) -> Pipeline:
    """The frozen marine model with the plan's rescaling rule; nothing is fitted."""
    m, name = copy.deepcopy(frozen), f"MARINE_{v}"
    for key, c in m.coef.items():
        m.coef[key] = {k: x * rate(name) / rate("MARINE") for k, x in c.items()}
    m.dist_coef = m.dist_coef * unit_stats(name)["speed"] / unit_stats("MARINE")["speed"]
    rho = min(pair_reach(name, name), m.max_reach)
    m.rho = {"marine": rho}
    for side in SIDES:
        p = np.array(m.P_team[f"share_{side}"], float)
        p[-1] = rho
        m.P_team[f"share_{side}"] = p
    return m


def firepower(m: Pipeline) -> dict:
    """k per losing side of a one-type model."""
    t = m.L.types[0]
    return {s: -m.coef[(s, t)][f"R[{t}]*s[{t}]"] for s in SIDES}


def as_marines(t: pd.DataFrame, v: str) -> pd.DataFrame:
    """Columns renamed for the marine model (health stays in ten full units of the variant)."""
    return t.rename(columns={c: c.replace(f"marine_{v.lower()}", "marine") for c in t})


def health_error(Y: np.ndarray, B: dict, h: float) -> float:
    n = round(h / H.DT)
    ok = ~np.isnan(B["Y"][:, n, 0])
    return float(np.abs(Y[ok, n, :2] - B["Y"][ok, n, :2]).mean())


def g0(n: int = 20) -> None:
    sampler, frames = CloseBattleSampler(), {}
    for scenario in ("10m_vs_10m", "10m_v0"):
        bounds = walkable_bounds(load_map_info(scenario))
        parts = []
        for seed in range(G0_SEED, G0_SEED + n):
            s = sampler.sample(seed, bounds)
            env = make_env(scenario, step_mul=STEP_MUL, spawn=s.spawn, unit_counts=s.unit_counts)
            ep = play_episode(
                env,
                IntentSchedule(withdraw_radius=7.0),
                seed=seed,
                max_steps=MAX_SECONDS * 16 // STEP_MUL,
                start_health=s.start_health,
            )
            parts.append(ep.snapshots.drop(columns="unit_type").assign(seed=seed))
        frames[scenario] = pd.concat(parts, ignore_index=True)
    pd.testing.assert_frame_equal(frames["10m_vs_10m"], frames["10m_v0"], check_exact=True)
    print(f"G0 passes: {n} battles, {len(frames['10m_v0'])} unit rows identical")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        path = OUT / f"{v.lower()}.parquet"
        if not path.exists():
            build_one(("exp13_properties", v.lower(), (f"MARINE_{v}",))).to_parquet(path)
            print("built", v, flush=True)


def halves(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = t.index.get_level_values(0) < SPLIT
    return t[first], t[~first]


def analyse() -> None:
    result = DOCS / "01_results.md"
    assert not result.exists(), "the analysis has already been run"
    frozen = pickle.load(open(H.CACHE / "generalisation_frozen.pkl", "rb"))["marines"]
    refits, coefs, errs, judged = {}, [], [], {}
    for v in VARIANTS:
        name = f"MARINE_{v}"
        fit_half, test_half = halves(pd.read_parquet(OUT / f"{v.lower()}.parquet"))
        own = Layout((name.lower(),))
        refits[v] = Pipeline((name,), withdraw_radius=7.0, **config()).fit(prepare(fit_half, own))
        models = {"unchanged": frozen, "rescaled": rescaled(frozen, v), "refit": refits[v]}
        st = add_controls(test_half)
        B_own, B = own.batch(st), frozen.L.batch(as_marines(st, v))
        np.testing.assert_allclose(B_own["team"]["X0"], B["team"]["X0"], atol=1e-12)
        T = B["team"]
        Y = {k: m.L.team_view(m.rollout(B_own if k == "refit" else B)) for k, m in models.items()}
        Y["persistence"] = np.repeat(T["X0"][:, None], T["U"].shape[1] + 1, 1)
        for k in ("persistence", "unchanged", "rescaled", "refit"):
            hm = H.horizon_metrics(Y[k], T)
            row = {"variant": v, "model": k} | {
                f"health_{h}s": health_error(Y[k], T, h) for h in (0.5, 1, 2, 4)
            }
            row |= {c: hm[c] for c in ("margin_r2_2s", "margin_r2_4s", "alive_2s", "distance_2s")}
            errs.append(row)
        judged[v] = {
            "ci": E.ci_health_diff(Y["unchanged"], Y["rescaled"], T, h=2),
            "ratio": health_error(Y["rescaled"], T, 2) / health_error(Y["refit"], T, 2),
        }
    k0, d0 = firepower(refits["V0"]), dict(zip(refits["V0"].dist_names, refits["V0"].dist_coef, strict=True))
    speed = []
    for v in VARIANTS:
        k, target = firepower(refits[v]), rate(f"MARINE_{v}") / rate("MARINE")
        for s in SIDES:
            t = refits[v].L.types[0]
            extra = [n for n, c in refits[v].coef[(s, t)].items() if c and not n.startswith("R[")]
            coefs.append(
                {
                    "variant": v,
                    "losing side": s,
                    "k refit": k[s],
                    "ratio to V0": k[s] / k0[s],
                    "rule": target,
                    "ratio / rule": k[s] / k0[s] / target,
                    "other terms kept": ", ".join(extra) or "-",
                }
            )
        if v in SPEED:
            d = dict(zip(refits[v].dist_names, refits[v].dist_coef, strict=True))
            f = unit_stats(f"MARINE_{v}")["speed"] / unit_stats("MARINE")["speed"]
            for n in DIST_P3:
                r = d[n] / d0[n] if d0[n] else np.nan
                speed.append({"variant": v, "term": n, "refit": d[n], "V0": d0[n], "ratio / rule": r / f})
    coefs, errs, speed = pd.DataFrame(coefs), pd.DataFrame(errs), pd.DataFrame(speed)
    ok = coefs["ratio / rule"].between(0.85, 1.15)
    n1, n2 = int(ok[coefs.variant.isin(FIREPOWER)].sum()), int(ok[coefs.variant.isin(SPEED + RANGE)].sum())
    n4 = sum(judged[v]["ci"][1] < 0 for v in FIREPOWER)
    n5 = sum(judged[v]["ratio"] <= 1.15 for v in FIREPOWER)
    rules = pd.DataFrame(
        {
            "result": [
                f"{n1} of 16 (needs 14)",
                f"{n2} of 8 (needs 7)",
                f"{int(speed['ratio / rule'].between(0.8, 1.2).sum())} of {len(speed)} (needs all)",
                f"{n4} of 8 (needs 8)",
                f"{n5} of 8 (needs 7)",
            ],
            "met": [n1 >= 14, n2 >= 7, bool(speed["ratio / rule"].between(0.8, 1.2).all()), n4 == 8, n5 >= 7],
        },
        index=[
            "P1 firepower",
            "P2 no effect",
            "P3 speed",
            "P4 rescaled beats unchanged",
            "P5 rescaled vs refit",
        ],
    )
    j = pd.DataFrame(judged).T
    j["ci"] = j["ci"].map(lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]")
    j.columns = ["rescaled - unchanged, health 2 s, paired 95%", "rescaled / refit, health 2 s"]
    lines = ["# Unit properties, one at a time: results", "", "Plan: `00_plan.md`. Script:"]
    lines += ["`experiments/properties/properties.py`. Run once.", "", "## Rules", "", md(rules), ""]
    lines += ["## Firepower (refits on 500 battles)", "", md(coefs.round(3)), ""]
    lines += ["## Speed (refits)", "", md(speed.round(3)), ""]
    lines += ["## Prediction (500 other battles; health in ten full units of the variant)", ""]
    lines += [md(j), "", md(errs.round(4)), "", "## Refitted equations", ""]
    for v in VARIANTS:
        lines += [f"{v}:", "", "```", equations(refits[v]), "```", ""]
    result.write_text("\n".join(lines))
    pickle.dump(refits, open(OUT / "refits.pkl", "wb"))
    print("\n".join(lines))


if __name__ == "__main__":
    {"g0": g0, "build": build, "analyse": analyse}[sys.argv[1]]()
