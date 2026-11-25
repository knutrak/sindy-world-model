# ruff: noqa: E501, E731, B007, B023, B905, I001
"""Shared helpers of the stress tests (docs/triple/04_stress.md). Train data only."""

import warnings

import numpy as np
from triple import config, rates

from sindy_wm.models.pipeline import OTHER, SIDES, Layout, Pipeline

warnings.simplefilter("ignore")
SC = {  # experiment, types, withdraw radius
    "marines": ("exp06_intents", ("MARINE",), 7.0),
    "marauders": ("exp09_marauders", ("MARAUDER",), 8.674),
    "stalkers": ("exp07_stalkers", ("STALKER",), 7.0 * 7.25 / 5.75),
    "mixed": ("exp08_mixed", ("MARINE", "MARAUDER"), 8.0),
    "triple": ("exp12_triple", ("MARINE", "MARAUDER", "STALKER"), 8.0),
}
_rows = {}


def load(name):
    if name not in _rows:
        e, ty, _ = SC[name]
        _rows[name] = rates(e, "train", Layout(tuple(t.lower() for t in ty)))
    return _rows[name]


def sub(name, n, seed=0):
    """n random training battles."""
    rows = load(name)
    eps = rows.index.get_level_values(0).unique().to_numpy()
    if n >= len(eps):
        return rows
    return rows.loc[np.sort(np.random.default_rng(seed).choice(eps, n, replace=False))]


def fit(name, rows, types=None, **kw):
    ty, wd = SC[name][1], SC[name][2]
    return Pipeline(types or ty, withdraw_radius=wd, **(config() | kw)).fit(rows)


def terms(m):
    return {(s, t, k): v for (s, t), c in m.coef.items() for k, v in c.items() if v}


def expected(m):
    return {(s, t, f"R[{a}]*s[{t}]") for s in SIDES for t in m.L.types for a in m.L.types}


def spurious(m):
    """(terms kept that are not firepower on the own type, firepower terms missing)."""
    k = set(terms(m))
    return len(k - expected(m)), len(expected(m) - k)


def head(t):
    print(f"\n### {t}", flush=True)


def features(m, rows, s, t):
    """Names and matrix of one health equation's candidates."""
    o = OTHER[s]
    Ro = {a: rows[f"reach_{o}_{a}"].to_numpy() for a in m.L.types}
    Ao = {a: rows[f"alive_{o}_{a}"].to_numpy() for a in m.L.types}
    F = m._terms(Ro, m._sigma_measured(rows, o), Ao, rows["withdraw"].to_numpy(), o, t)
    return list(F), np.column_stack(list(F.values()))
