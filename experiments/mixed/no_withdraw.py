"""What if the withdraw intent were removed? An imitation with existing data (20 Nov, exploratory).

About 45% of marine battles (exp06) and 33% of mixed battles (exp08, reach 6) never use withdraw: the
intent schedule simply never drew it. Fitting only on those battles imitates data collected without the
intent. Every model is scored on the val battles that also never withdraw (and, for reference, on all val).

    fitted on      scored on          meaning
    all train      all val            the current models
    all train      no-withdraw val    the current models, on battles without withdraw
    no-wd train    no-withdraw val    "withdraw removed": fitted and used without the intent

Caveat: battles without withdraw are not a random sample: short battles have fewer intent segments, so
they are over-represented. The table reports battle length so this is visible. A clean answer needs a new
collection with movement weights (advance, hold, withdraw) = (0.625, 0.375, 0).

    uv run --extra explore experiments/mixed/no_withdraw.py      -> docs/mixed/06_no_withdraw.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle  # noqa: E402
from rollouts import team_rates  # noqa: E402

from sindy_wm.evaluation import harness as H  # noqa: E402
from sindy_wm.evaluation.explore_models import AliveVariantA  # noqa: E402
from sindy_wm.evaluation.harness_models import closures_for  # noqa: E402
from sindy_wm.models.intent_model import (
    add_controls,  # noqa: E402
    fit_derived_share,  # noqa: E402
)
from sindy_wm.models.typed_model import (  # noqa: E402
    TypedA,
    team_battle_batch,
    team_horizon_batch,
    team_view,
    typed_batch,
)
from sindy_wm.paths import PROJECT_ROOT  # noqa: E402

DOC = PROJECT_ROOT / "docs" / "mixed" / "06_no_withdraw.md"
COLS = [
    "health_2s",
    "health_4s",
    "margin_r2_2s",
    "margin_r2_4s",
    "alive_4s",
    "distance_4s",
    "battle_health_nrmse",
    "winner_acc",
    "duration_err_s",
    "n_battles",
]


def md(table) -> str:
    return "```\n" + table.to_string() + "\n```"


def no_wd_episodes(states: pd.DataFrame) -> np.ndarray:
    wd = (states["movement"] == "withdraw").groupby(level=0).sum()
    return wd.index[wd == 0].to_numpy()


def terms(m) -> dict:
    t = {
        f"{eq}: {n}": c
        for eq in ("health_blue", "health_red", "distance")
        for n, c in zip(m.names, m.coef[eq], strict=True)
        if c
    }
    return t | {f"alive_{s}: {k}": v for s in ("blue", "red") for k, v in m.p[s].items() if v}


def closures_without_withdraw(all_rates: pd.DataFrame, rates: pd.DataFrame) -> dict:
    """Closures for a fit on no-withdraw battles: reach curves refitted on them; the movement gates from all
    battles (the withdraw gate can't be fitted without withdraw steps, and is never used without them)."""
    P = closures_for(H.FitData(states=None, rates=all_rates))
    P["share_blue"] = fit_derived_share(rates, "blue")[0]
    P["share_red"] = fit_derived_share(rates, "red")[0]
    return P


def marines() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_states, va = H.load_states("train"), H.load_states("val")
    rates = H.rows_with_rates("train")
    keep_tr, keep_va = no_wd_episodes(tr_states), no_wd_episodes(va)
    fits = {
        "all train": AliveVariantA(law="reach").fit(H.FitData(states=None, rates=H.both_alive(rates))),
        "no-wd train": AliveVariantA(
            law="reach",
            P=closures_without_withdraw(H.both_alive(rates), H.both_alive(H.select(rates, keep_tr))),
        ).fit(H.FitData(states=None, rates=H.both_alive(H.select(rates, keep_tr)))),
    }
    rows = {}
    for scored, v in (("all val", va), ("no-wd val", H.select(va, keep_va))):
        Bh, Bb = H.horizon_batch(v), H.battle_batch(v)
        for fitted, m in fits.items():
            if fitted == "no-wd train" and scored == "all val":
                continue
            rows[(f"fitted on {fitted}", f"scored on {scored}")] = {
                **H.horizon_metrics(m.rollout(Bh), Bh),
                **H.battle_metrics(m.rollout(Bb), Bb),
                "mean battle s": float(v.groupby(level=0)["t"].max().mean()),
            }
    coefs = pd.DataFrame({k: terms(m.m) for k, m in fits.items()})
    return pd.DataFrame(rows).T, coefs


def mixed() -> tuple[pd.DataFrame, pd.DataFrame]:
    oracle.configure(6.0)
    tr_rates = oracle.rates("train")
    va = add_controls(pd.read_parquet(oracle.P / "val.parquet"))
    tr_states = pd.read_parquet(oracle.P / "train.parquet")
    keep_tr, keep_va = no_wd_episodes(tr_states), no_wd_episodes(va)
    fits = {}
    for fitted, r in (("all train", tr_rates), ("no-wd train", H.select(tr_rates, keep_tr))):
        P = None if fitted == "all train" else closures_without_withdraw(team_rates(tr_rates), team_rates(r))
        team = AliveVariantA(law="reach", P=P).fit(H.FitData(states=None, rates=team_rates(r)))
        typed = TypedA(team, max_reach=6.0, withdraw_rule=True).fit(r)
        fits[fitted] = {"team A*-2k": team, "typed + wd rule": typed}
    rows = {}
    for scored, v in (("all val", va), ("no-wd val", H.select(va, keep_va))):
        Bh = typed_batch(v)
        Th = team_horizon_batch(Bh)
        Bb, Tb = team_battle_batch(v)
        for fitted, ms in fits.items():
            if fitted == "no-wd train" and scored == "all val":
                continue
            for name, m in ms.items():
                if name.startswith("typed"):
                    Yh, Yb = team_view(m.rollout(Bh)), team_view(m.rollout(Bb))
                else:
                    Yh, Yb = m.rollout(Th), m.rollout(Tb)
                rows[(name, f"fitted on {fitted}", f"scored on {scored}")] = {
                    **H.horizon_metrics(Yh, Th),
                    **H.battle_metrics(Yb, Tb),
                    "mean battle s": float(v.groupby(level=0)["t"].max().mean()),
                }
    coefs = pd.DataFrame({k: terms(ms["team A*-2k"].m) for k, ms in fits.items()})
    return pd.DataFrame(rows).T, coefs


def main() -> None:
    m_tab, m_coef = marines()
    x_tab, x_coef = mixed()
    cols = COLS + ["mean battle s"]
    lines = [
        "# What if withdraw were removed? (imitation with existing data, 20 Nov)",
        "",
        "Script: `experiments/mixed/no_withdraw.py`. Battles that never withdraw: marines 45% (train 2686,",
        "val 455), mixed 33% (train 2006, val 329). See the script's docstring for the caveat.",
        "",
        "## Marines (exp06), A*-2k",
        "",
        md(m_tab[cols].round(3)),
        "",
        "Coefficients:",
        "",
        md(m_coef.round(4)),
        "",
        "## Mixed (exp08, reach 6)",
        "",
        md(x_tab[cols].round(3)),
        "",
        "Team model coefficients:",
        "",
        md(x_coef.round(4)),
        "",
    ]
    DOC.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
