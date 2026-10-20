
"""Clean Matplotlib dashboard for comparing Q-Learning and SARSA."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

import config


PANELS = ("reward", "success", "steps", "summary")

PANEL_LABELS = {
    "reward": "Learning Curve",
    "success": "Success Rate",
    "steps": "Steps per Success",
    "summary": "Performance Summary",
}

# Dashboard colors
BG = "#070b1e"
AX_BG = "#101936"
GRID = "#29385d"
TEXT = "#edf2ff"
MUTED = "#a7b5d6"
ACCENT = "#55d6ff"


def rolling_mean(values, window):
    """Calculate a trailing rolling average."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    window = max(1, min(int(window), len(values)))
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    indices = np.arange(len(values))
    low = np.maximum(0, indices - window + 1)

    return (
        cumulative[indices + 1] - cumulative[low]
    ) / (indices + 1 - low)


def rolling_steps_per_success(steps, success, window):
    """Average steps over successful episodes in each rolling window."""
    steps = np.asarray(steps, dtype=float)
    success = np.asarray(success, dtype=float)

    if len(steps) == 0:
        return steps

    window = max(1, min(int(window), len(steps)))

    cumulative_steps = np.concatenate(
        ([0.0], np.cumsum(steps * success))
    )
    cumulative_success = np.concatenate(
        ([0.0], np.cumsum(success))
    )

    indices = np.arange(len(steps))
    low = np.maximum(0, indices - window + 1)

    total_steps = (
        cumulative_steps[indices + 1] - cumulative_steps[low]
    )
    success_count = (
        cumulative_success[indices + 1] - cumulative_success[low]
    )

    result = np.full(len(steps), np.nan, dtype=float)
    valid = success_count > 0
    result[valid] = total_steps[valid] / success_count[valid]

    return result


def _style_axes(ax, title, xlabel, ylabel, fontsize):
    """Apply consistent formatting to each graph."""
    ax.set_facecolor(AX_BG)

    ax.set_title(
        title,
        color=TEXT,
        fontsize=fontsize + 1,
        fontweight="bold",
        pad=9,
    )

    ax.set_xlabel(xlabel, color=MUTED, fontsize=fontsize)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=fontsize)

    ax.tick_params(
        axis="both",
        colors=MUTED,
        labelsize=max(7, fontsize - 1),
    )

    ax.grid(
        True,
        color=GRID,
        alpha=0.65,
        linewidth=0.7,
        linestyle="--",
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color(GRID)


def _legend(ax, fontsize):
    legend = ax.legend(
        fontsize=max(7, fontsize - 1),
        facecolor=BG,
        edgecolor=GRID,
        loc="best",
        framealpha=0.95,
    )

    if legend:
        for label in legend.get_texts():
            label.set_color(TEXT)


def _plot_curve(
    ax,
    data,
    transform,
    title,
    ylabel,
    fontsize,
    ylim=None,
):
    """Plot the rolling performance curves of available models."""
    drew_anything = False

    for algorithm in config.ALGORITHM_KEYS:
        if algorithm not in data:
            continue

        history = data[algorithm]["history"]
        values = transform(history)

        if len(values) == 0:
            continue

        episodes = np.arange(1, len(values) + 1)
        color = config.ALGORITHM_COLORS[algorithm]
        name = config.ALGORITHM_NAMES[algorithm]

        # The thicker line makes the learning trend easy to see.
        ax.plot(
            episodes,
            values,
            color=color,
            linewidth=2.0,
            label=name,
            solid_capstyle="round",
        )

        drew_anything = True

    _style_axes(
        ax,
        title,
        "Training Episode",
        ylabel,
        fontsize,
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    if drew_anything:
        _legend(ax, fontsize)
    else:
        ax.text(
            0.5,
            0.5,
            "No training history available",
            color=MUTED,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


def _plot_summary(ax, data, fontsize):
    """Display evaluation metrics in a comparison table."""
    ax.set_facecolor(AX_BG)
    ax.axis("off")

    ax.set_title(
        "Performance Summary — Independent Evaluation",
        color=TEXT,
        fontsize=fontsize + 1,
        fontweight="bold",
        pad=12,
    )

    columns = [
        "Algorithm",
        "Episodes",
        "Mean Reward",
        "Success",
        "Steps / Success",
        "Collision",
        "Mean Crystals",
    ]

    rows = []

    for algorithm in config.ALGORITHM_KEYS:
        if algorithm not in data:
            continue

        entry = data[algorithm]
        evaluation = entry["evaluation"]

        mean_steps = evaluation.get("mean_steps_success")

        rows.append([
            config.ALGORITHM_NAMES[algorithm],
            str(entry["episodes"]),
            f"{evaluation['mean_reward']:.2f}",
            f"{evaluation['success_rate']:.1f}%",
            "N/A" if mean_steps is None else f"{mean_steps:.2f}",
            f"{evaluation['collision_rate']:.1f}%",
            f"{evaluation['mean_crystals']:.2f}",
        ])

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No trained models available.\nTrain an agent first.",
            color=MUTED,
            ha="center",
            va="center",
            fontsize=fontsize + 1,
            transform=ax.transAxes,
        )
        return

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.01, 0.25, 0.98, 0.52],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(max(6, fontsize - 1))

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        cell.set_text_props(color=TEXT)

        if row == 0:
            cell.set_facecolor("#23345e")
            cell.set_text_props(
                color=ACCENT,
                weight="bold",
            )
        else:
            cell.set_facecolor(
                "#172442" if row % 2 else AX_BG
            )

    first_entry = next(iter(data.values()))
    evaluation = first_entry["evaluation"]

    evaluation_episodes = evaluation.get("episodes", "configured")
    noise = evaluation.get("noise")

    if noise is None:
        footer = (
            f"Evaluation episodes: {evaluation_episodes} | "
            f"Fixed seed: {config.EVAL_SEED}"
        )
    else:
        footer = (
            f"Evaluation episodes: {evaluation_episodes} | "
            f"Random-move noise: {noise * 100:.0f}% | "
            f"Fixed seed: {config.EVAL_SEED}"
        )

    ax.text(
        0.5,
        0.12,
        footer,
        color=MUTED,
        ha="center",
        va="center",
        fontsize=max(6, fontsize - 1),
        transform=ax.transAxes,
    )

    ax.text(
        0.5,
        0.04,
        "Higher reward and success are better; fewer steps and collisions are better.",
        color=MUTED,
        ha="center",
        va="center",
        fontsize=max(6, fontsize - 1),
        transform=ax.transAxes,
    )


def build_figure(
    data,
    which="all",
    figsize=(10.0, 5.4),
    dpi=100,
    window=None,
):
    """Build the complete dashboard or one selected graph."""
    window = (
        config.ROLLING_WINDOW
        if window is None
        else window
    )

    fig = Figure(
        figsize=figsize,
        dpi=dpi,
        facecolor=BG,
    )

    FigureCanvasAgg(fig)

    all_panels = which == "all"
    fontsize = 8 if all_panels else 11

    panels = PANELS if all_panels else (which,)

    for index, panel in enumerate(panels, start=1):
        if all_panels:
            ax = fig.add_subplot(2, 2, index)
        else:
            ax = fig.add_subplot(1, 1, 1)

        if panel == "reward":
            _plot_curve(
                ax,
                data,
                lambda history: rolling_mean(
                    history["reward"], window
                ),
                f"Graph 1: Learning Curve (window {window})",
                "Mean Episode Reward",
                fontsize,
            )

        elif panel == "success":
            _plot_curve(
                ax,
                data,
                lambda history: 100.0 * rolling_mean(
                    history["success"], window
                ),
                f"Graph 2: Success Rate (window {window})",
                "Success Rate (%)",
                fontsize,
                ylim=(0, 100),
            )

        elif panel == "steps":
            _plot_curve(
                ax,
                data,
                lambda history: rolling_steps_per_success(
                    history["steps"],
                    history["success"],
                    window,
                ),
                f"Graph 3: Steps per Successful Mission (window {window})",
                "Mean Steps per Success",
                fontsize,
            )

        elif panel == "summary":
            _plot_summary(ax, data, fontsize)

        else:
            raise ValueError(
                f"Unknown panel: {panel}. Choose from {PANELS} or 'all'."
            )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.91,
        bottom=0.14,
        hspace=0.43,
        wspace=0.32,
    )

    return fig


def figure_to_rgba(fig):
    """Return the figure as RGBA bytes and its pixel dimensions."""
    canvas = (
        fig.canvas
        if isinstance(fig.canvas, FigureCanvasAgg)
        else FigureCanvasAgg(fig)
    )

    canvas.draw()

    return bytes(canvas.buffer_rgba()), canvas.get_width_height()


def save_comparison_png(data, path, window=None):
    """Save the complete comparison dashboard as a PNG image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure(
        data,
        which="all",
        figsize=(14.0, 8.0),
        dpi=160,
        window=window,
    )

    fig.savefig(
        path,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        dpi=160,
    )

    return path
