"""Shared helpers for the generalisation experiments (docs/generalisation/00_plan.md)."""

from pathlib import Path

import pandas as pd

from sindy_wm.models.pipeline import Layout, prepare
from sindy_wm.paths import DATA_DIR, PROJECT_ROOT

DOCS = PROJECT_ROOT / "docs" / "generalisation"
SCENARIOS = {
    "marines": ("exp06_intents", ("MARINE",)),
    "marauders": ("exp09_marauders", ("MARAUDER",)),
    "mixed": ("exp08_mixed", ("MARINE", "MARAUDER")),
}


def folder(scenario: str) -> Path:
    return DATA_DIR / "processed" / f"{SCENARIOS[scenario][0]}_typed"


def layout(scenario: str) -> Layout:
    return Layout(tuple(t.lower() for t in SCENARIOS[scenario][1]))


def states(scenario: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(folder(scenario) / f"{name}.parquet")


def rows(scenario: str, name: str = "train") -> pd.DataFrame:
    """Rates and controls (cached next to the tables)."""
    path = folder(scenario) / f"rates_{name}_w9.parquet"
    if path.exists():
        return pd.read_parquet(path)
    r = prepare(states(scenario, name), layout(scenario))
    r.to_parquet(path)
    return r


def md(table) -> str:
    return "```\n" + table.to_string() + "\n```"
