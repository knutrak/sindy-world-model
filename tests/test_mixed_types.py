"""Two unit types in one battle: the unit-type override and the mixed sampler (branch mixed-marauders)."""

import pytest

from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.episode_sampling import CloseBattleSampler, MixedBattleSampler
from sindy_wm.envs.intents import IntentSchedule
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds

BOUNDS = walkable_bounds(load_map_info("mixed_m_mrd"))


def test_unit_types_override_places_the_given_types():
    types = (("MARINE", "MARAUDER", "MARINE"), ("MARAUDER", "MARAUDER", "MARINE", "MARINE"))
    env = make_env("mixed_m_mrd", step_mul=2, unit_types=types)
    episode = play_episode(env, IntentSchedule(withdraw_radius=8.0), seed=0, max_steps=2)
    first = episode.snapshots[episode.snapshots["step"] == 0]
    counts = first.groupby(["team", "unit_type"]).size().to_dict()
    assert counts == {
        ("blue", "MARAUDER"): 1,
        ("blue", "MARINE"): 2,
        ("red", "MARAUDER"): 2,
        ("red", "MARINE"): 2,
    }
    assert set(first.loc[first["unit_type"] == "MARAUDER", "max_hp"]) == {125.0}


def test_unit_types_and_unit_counts_are_exclusive():
    with pytest.raises(ValueError):
        make_env("mixed_m_mrd", unit_counts=(3, 3), unit_types=(("MARINE",), ("MARINE",)))


def test_mixed_sampler_keeps_close_battle_draws_and_one_share_for_both_sides():
    mixed, close = MixedBattleSampler(), CloseBattleSampler(unit_size=1.125)
    for seed in range(40):
        m, c = mixed.sample(seed, BOUNDS), close.sample(seed, BOUNDS)
        assert m.unit_counts == c.unit_counts and m.start_health == c.start_health and m.spawn == c.spawn
        assert tuple(len(t) for t in m.unit_types) == m.unit_counts
        shares = [t.count("MARAUDER") / len(t) for t in m.unit_types]
        assert all(1 <= t.count("MARAUDER") <= len(t) - 1 for t in m.unit_types)
        assert abs(shares[0] - shares[1]) < 0.2  # the same drawn share, rounded per side
        assert m == mixed.sample(seed, BOUNDS)  # deterministic per seed
        rec = m.record()
        assert rec["n_marauder_blue"] + rec["n_marine_blue"] == rec["n_blue"]
