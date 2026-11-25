"""Custom unit types (configs/units/): found and recorded by name, and V0 equal to the built-in marine."""

import json

import pytest
from smaclite.env.units.unit_type import StandardUnit

from sindy_wm.envs import units
from sindy_wm.envs.scenarios import make_env
from sindy_wm.envs.smaclite_snapshot import SnapshotLogger
from sindy_wm.paths import UNIT_DIR
from sindy_wm.states.typed import pair_reach, unit_stats

FIELDS = (
    "hp",
    "armor",
    "damage",
    "cooldown",
    "speed",
    "attack_range",
    "size",
    "shield",
    "minimum_scan_range",
)
FIELDS += ("attributes", "valid_targets", "bonuses", "plane", "attacks", "combat_type", "hp_regen")


def test_v0_has_the_marines_statistics():
    v0, marine = units.stats("MARINE_V0"), StandardUnit.MARINE.stats
    for f in FIELDS:
        assert getattr(v0, f) == getattr(marine, f), f


@pytest.mark.parametrize("name", [n for n in units.register() if n[7:] not in ("V0", "N1", "N2", "N3")])
def test_each_variant_changes_exactly_one_property(name):
    v0 = json.loads((UNIT_DIR / "MARINE_V0.json").read_text())
    v = json.loads((UNIT_DIR / f"{name}.json").read_text())
    assert sum(v[k] != v0[k] for k in v0) == 1 and set(v) == set(v0)


@pytest.mark.parametrize("name", units.register())
def test_scenario_loads_from_any_folder_and_is_recorded_by_name(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = make_env(f"10m_{name.removeprefix('MARINE_').lower()}", step_mul=2)
    env.reset(seed=1)
    snap = SnapshotLogger(env).start_episode()
    assert set(snap["unit_type"]) == {name} and len(snap) == 20
    assert set(snap["max_hp"]) == {unit_stats(name)["hp"]}


def test_reach_follows_the_range_and_unknown_units_are_refused():
    assert pair_reach("MARINE_R4", "MARINE_R4") == pytest.approx(4.75)
    assert pair_reach("MARINE", "MARINE") == pytest.approx(5.75)
    with pytest.raises(KeyError):
        unit_stats("NOT_A_UNIT")


def test_built_in_names_are_unchanged():
    env = make_env("10m_vs_10m", step_mul=2)
    env.reset(seed=1)
    assert set(SnapshotLogger(env).start_episode()["unit_type"]) == {"MARINE"}
