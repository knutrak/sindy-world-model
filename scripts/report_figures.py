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


def attack_ranges() -> None:
    """The attack rules of SMAClite drawn to scale for marine on marine and marauder on marauder.

    Weapon: centre distance <= range + both radii (Unit.has_within_attack_range); radius = size / 2.
    Blue's orders: target centre within AGENT_TARGET_RANGE = 6.
    """
    from matplotlib.patches import Circle

    from sindy_wm.states.typed import unit_stats

    limit = 6.0
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6), constrained_layout=True)
    for ax, kind in zip(axes, ("MARINE", "MARAUDER"), strict=True):
        s = unit_stats(kind)
        r, rng = s["radius"], s["range"]
        weapon = rng + 2 * r
        blue = min(weapon, limit)
        ax.add_patch(Circle((0, 0), blue, fc="#dfe7fb", ec="none", zorder=0))
        ax.add_patch(Circle((0, 0), weapon, fill=False, ec=BLUE, lw=1.4))
        ax.add_patch(Circle((0, 0), limit, fill=False, ec=GREY, lw=1.1, ls="--"))
        ax.add_patch(Circle((0, 0), r, fc=BLUE, ec="none"))
        ax.add_patch(Circle((weapon, 0), r, fc=RED, ec="none", alpha=0.9))
        # the weapon reach split into its parts, along the line to the target
        y = -1.0
        parts = ((0, r, "r"), (r, r + rng, f"range {rng:g}"), (r + rng, weapon, "r"))
        for a, b, txt in parts:
            ax.annotate(
                "",
                (a, y),
                (b, y),
                arrowprops={
                    "arrowstyle": "|-|, widthA=0.25, widthB=0.25",
                    "lw": 0.8,
                    "color": GREY,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
            ax.text((a + b) / 2, y - 0.25, txt, ha="center", va="top", fontsize=7, color=GREY)
        ax.text(0, r + 0.25, "attacker", ha="center", va="bottom", fontsize=7, color=BLUE)
        ax.text(weapon, r + 0.25, "target", ha="center", va="bottom", fontsize=7, color=RED)
        name = kind.capitalize()
        which = "weapon" if weapon <= limit else "limit of 6"
        ax.set_title(f"{name} on {name.lower()}\nreach {blue:g}, set by the {which}", fontsize=8)
        ax.set_xlim(-1.0, 7.9)
        ax.set_ylim(-2.4, 7.9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Map units", fontsize=8)
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.4),
        plt.Line2D([], [], color=GREY, lw=1.1, ls="--"),
        plt.Rectangle((0, 0), 1, 1, fc="#dfe7fb", ec="none"),
    ]
    labels = [
        "weapon: range + both radii",
        "limit on attack orders (6)",
        "reach",
    ]
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "env_attack_ranges.pdf")


def battle_frame(episode: int = 2, steps: tuple[int, int] = (0, 24)) -> None:
    """Two frames of one three-type training battle (exp12): units as circles of their real size, by side.

    Marines are light, marauders solid, stalkers solid with a thick light ring (their shield).
    A line goes from each unit to a target it has in range.
    """
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Circle

    from sindy_wm.io import storage
    from sindy_wm.paths import RAW_DIR
    from sindy_wm.states.typed import unit_stats

    s = storage.load_snapshots(RAW_DIR / "exp12_triple" / "train_0", [episode])
    s = s[s["alive"]]
    team = {"blue": BLUE, "red": RED}
    light = {"blue": "#a9bcf0", "red": "#e8aaa3"}

    def style(kind, side):
        if kind == "MARINE":
            return {"fc": light[side], "ec": team[side], "lw": 0.8}
        if kind == "MARAUDER":
            return {"fc": team[side], "ec": team[side], "lw": 0.8}
        return {"fc": team[side], "ec": light[side], "lw": 2.2}

    box = s[s["step"].isin(steps)]
    x0, x1 = box["x"].min() - 1.5, box["x"].max() + 1.5
    y0, y1 = box["y"].min() - 1.5, box["y"].max() + 1.5
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4), constrained_layout=True)
    for ax, k in zip(axes, steps, strict=True):
        f = s[s["step"] == k].set_index(["team", "unit_id"])
        for (side, _), u in f.iterrows():
            if u["target_in_range"] and u["target_team"] in team:
                t = f.loc[(u["target_team"], u["target_id"])]
                ax.plot([u["x"], t["x"]], [u["y"], t["y"]], color=team[side], lw=0.6, alpha=0.7, zorder=1)
        for (side, _), u in f.iterrows():
            r = unit_stats(u["unit_type"])["radius"]
            ax.add_patch(Circle((u["x"], u["y"]), r, zorder=2, **style(u["unit_type"], side)))
        n = f.groupby(level=0).size()
        ax.set_title(f"t = {k * 0.125:g} s: Blue {n.get('blue', 0)}, Red {n.get('red', 0)} units", fontsize=8)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Map units", fontsize=8)

    def mark(kind, side, ms):
        st = style(kind, side)
        return plt.Line2D([], [], marker="o", ls="", ms=ms, mfc=st["fc"], mec=st["ec"], mew=st["lw"])

    handles = [
        (mark("MARINE", "blue", 5), mark("MARINE", "red", 5)),
        (mark("MARAUDER", "blue", 6.5), mark("MARAUDER", "red", 6.5)),
        (mark("STALKER", "blue", 7), mark("STALKER", "red", 7)),
        (plt.Line2D([], [], color=BLUE, lw=0.8), plt.Line2D([], [], color=RED, lw=0.8)),
    ]
    fig.legend(
        handles,
        ["marine", "marauder", "stalker (ring: shield)", "attacking a target in reach"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
        loc="outside lower center",
        ncol=4,
        fontsize=7.5,
        frameon=False,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "env_battle_frame.pdf")


if __name__ == "__main__":
    example_battle()
    attack_ranges()
    battle_frame()
