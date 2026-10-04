"""Plotting functions for analysing data and evaluating models."""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

TEAM_COLORS = {"blue": "tab:blue", "red": "tab:red"}


def plot_level_a(states: pd.DataFrame, title: str = "") -> plt.Figure:
    """Plot the Level A state of one episode.

    Args:
        states: Level A state of one episode, as returned by `level_a()`.
        title: Title of the figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    colors = [TEAM_COLORS["blue"], TEAM_COLORS["red"]]
    states[["health_blue", "health_red"]].plot(ax=axes[0], ylabel="total hp", color=colors)
    states[["alive_blue", "alive_red"]].plot(ax=axes[1], ylabel="alive units", color=colors)
    states["distance"].plot(ax=axes[2], ylabel="centroid distance", xlabel="step", color="k")
    axes[0].set_title(title)
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    return fig


def _setup_map(snapshots: pd.DataFrame, figsize=(6, 6)):
    """Figure with fixed axis limits covering all positions in the episode."""
    fig, ax = plt.subplots(figsize=figsize)
    margin = 2
    ax.set_xlim(snapshots["x"].min() - margin, snapshots["x"].max() + margin)
    ax.set_ylim(snapshots["y"].min() - margin, snapshots["y"].max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    return fig, ax


def plot_positions(snapshots: pd.DataFrame, step: int, title: str = "") -> plt.Figure:
    """Plot the positions of all alive units at one step of an episode.

    Args:
        snapshots: Per-unit snapshot table of one episode.
        step: The step for which to plot positions.
        title: Title of the figure.
    """
    fig, ax = _setup_map(snapshots)
    at_step = snapshots[(snapshots["step"] == step) & snapshots["alive"]]
    for team, color in TEAM_COLORS.items():
        units = at_step[at_step["team"] == team]
        ax.scatter(units["x"], units["y"], color=color, label=team)
    ax.set_title(title or f"step {step}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def animate_positions(
    snapshots: pd.DataFrame, interval: int = 300, title: str = ""
) -> FuncAnimation:
    """Animate the positions of all alive units over an episode.

    Args:
        snapshots: Per-unit snapshot table of one episode.
        interval: Milliseconds between frames.
        title: Text shown before the step counter in the title.

    Returns:
        The animation. In a notebook, show it with
        `IPython.display.HTML(anim.to_jshtml())`, or save it with
        `anim.save("episode.gif", writer="pillow")`.
    """
    fig, ax = _setup_map(snapshots)
    steps = sorted(snapshots["step"].unique())
    by_step = {step: frame for step, frame in snapshots.groupby("step")}

    # One scatter per team; each frame only moves the points
    scatters = {
        team: ax.scatter([], [], color=color, label=team) for team, color in TEAM_COLORS.items()
    }
    ax.legend(loc="upper right")

    def update(step):
        frame = by_step[step]
        for team, scatter in scatters.items():
            units = frame[(frame["team"] == team) & frame["alive"]]
            scatter.set_offsets(units[["x", "y"]].to_numpy().reshape(-1, 2))
        n_blue = int(frame.loc[frame["team"] == "blue", "alive"].sum())
        n_red = int(frame.loc[frame["team"] == "red", "alive"].sum())
        ax.set_title(f"{title}  step {step}   blue {n_blue} vs red {n_red}".strip())
        return list(scatters.values())

    anim = FuncAnimation(fig, update, frames=steps, interval=interval)
    plt.close(fig)  # stop notebooks from also showing an empty static figure
    return anim