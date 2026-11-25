"""The world-model interface in one battle: observe, ask what if, compare with what the game then does.

    uv run scripts/world_model_demo.py            marines
    uv run scripts/world_model_demo.py mixed 7    the mixed model, battle seed 7

One battle illustrates the calls; it is not evidence of accuracy (docs/world_model.md).
"""

import sys

from sindy_wm.envs.episode_sampling import CloseBattleSampler, MixedBattleSampler
from sindy_wm.envs.intents import IntentPolicy
from sindy_wm.envs.scenarios import load_map_info, make_env, walkable_bounds
from sindy_wm.models.intent_model import DT
from sindy_wm.world_model import WorldModel

SETUPS = {
    "marines": ("10m_vs_10m", CloseBattleSampler()),
    "marauders": ("10mrd_vs_10mrd", CloseBattleSampler(unit_size=1.125)),
    "mixed": ("mixed_m_mrd", MixedBattleSampler()),
}
STEP_MUL = 2  # 0.125 s per step, the model's own step


def main(name: str = "marines", seed: int = 3) -> None:
    wm = WorldModel.load(name)
    print(wm, "\n\n" + wm.equations() + "\n")

    scenario, sampler = SETUPS[name]
    s = sampler.sample(seed, walkable_bounds(load_map_info(scenario)))
    types = getattr(s, "unit_types", None)
    env = make_env(
        scenario,
        step_mul=STEP_MUL,
        spawn=s.spawn,
        unit_counts=None if types else s.unit_counts,
        unit_types=types,
    )
    env.reset(seed=seed)
    policy = IntentPolicy("advance", "focus", withdraw_radius=wm.pipeline.withdraw_radius)
    policy.reset(seed)
    while wm.observe(env).in_reach("blue") == 0:  # walk until the first contact
        env.step(policy.act(env))

    state = wm.observe(env)
    print(state.to_text() + "\n")
    print(wm.compare(state).round(1).to_string(index=False) + "\n")

    prediction = wm.predict(state, "advance/focus", horizon=wm.reliable_horizon)
    print(prediction.to_text() + "\n")
    over = False
    for _ in range(round(wm.reliable_horizon / DT)):
        _, _, terminated, truncated, _ = env.step(policy.act(env))
        over = terminated or truncated
        if over:
            break
    print("The battle ended first." if over else "What the game did:\n" + wm.observe(env).to_text())


if __name__ == "__main__":
    main(*sys.argv[1:2], *(int(a) for a in sys.argv[2:3]))
