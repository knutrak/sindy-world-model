"""Figures for the report (Overleaf project in report/): written to report/figures/.

uv run scripts/report_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sindy_wm.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "report" / "figures"
BLUE, RED, GREY = "#2f5bd3", "#c0392b", "#444444"
SHADE = {"advance": "#ffffff", "hold": "#e9e9e9", "withdraw": "#f6dfb8"}


def example_battle() -> None:
    """One marine training battle, by rule: the first of median duration that uses all three movements."""
    t = pd.read_parquet(DATA_DIR / "processed" / "exp06_intents_typed" / "train.parquet")
    g = t.groupby(level=0)
    length = g["t"].max()
    every = g["movement"].nunique() == 3
    ep = length[(length == length.median()) & every].index[0]
    b = t.loc[ep]
    print("battle", ep, "duration", b["t"].max())
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.2), sharex=True, constrained_layout=True)
    quantities = (
        ("health", "Health (ten full units)", 1),
        ("alive", "Units alive", 10),
        ("reach", "Units in reach", 10),
    )
    for ax, (name, label, f) in zip(axes.flat, quantities, strict=False):
        for side, c in (("blue", BLUE), ("red", RED)):
            ax.plot(b["t"], f * b[f"{name}_{side}_marine"], color=c, lw=1.4, label=side.capitalize())
        ax.set_ylabel(label, fontsize=8)
    axes[1, 1].plot(b["t"], b["distance"], color=GREY, lw=1.4)
    axes[1, 1].set_ylabel("Distance (map units)", fontsize=8)
    change = b["movement"].ne(b["movement"].shift()).cumsum()
    for ax in axes.flat:
        for _, s in b.groupby(change):
            ax.axvspan(
                s["t"].iloc[0], s["t"].iloc[-1] + 0.125, color=SHADE[s["movement"].iloc[0]], lw=0, zorder=0
            )
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, b["t"].max())
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)", fontsize=8)
    patches = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#999999", lw=0.5) for c in SHADE.values()]
    lines = axes[0, 0].get_lines()
    fig.legend(
        [*lines, *patches],
        ["Blue", "Red", *SHADE],
        loc="outside upper center",
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "env_example_battle.pdf")


if __name__ == "__main__":
    example_battle()
