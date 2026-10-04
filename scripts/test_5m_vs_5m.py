from sindy_wm.data.storage import save_episode
from sindy_wm.envs.episode import play_episode
from sindy_wm.envs.policies import RandomPolicy
from sindy_wm.envs.scenarios import make_env
from sindy_wm.paths import RAW_DIR


def main():

    env = make_env("5m_vs_5m")
    episode = play_episode(env, RandomPolicy(), seed=0)
    save_episode(
        episode.snapshots, dataset_dir=RAW_DIR / "tests", episode_idx=0, overwrite=True
    )
    print(f"{episode.snapshots.step.max()} steps, {len(episode.snapshots)} snapshot rows")
    print(episode.snapshots.groupby(["step", "team"])["hp"].sum().unstack().tail())
    print("battle_won:", episode.battle_won, "truncated:", episode.truncated)


if __name__ == "__main__":
    main()
