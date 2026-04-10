from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt

GRID_COLOR = "#D5DDE5"
AXIS_COLOR = "#AAB7C4"
PLOT_BG_COLOR = "#F8FAFC"
DEFAULT_BAR_COLOR = "#2F6CAD"


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PLOT_BG_COLOR)
    axis.grid(axis="y", color=GRID_COLOR, alpha=0.9, linestyle="--", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(AXIS_COLOR)
    axis.spines["bottom"].set_color(AXIS_COLOR)


def resolve_summary_fig_width(labels: list[str]) -> float:
    if not labels:
        return 8.0
    label_len = max(len(label) for label in labels)
    dynamic = 4.8 + len(labels) * 1.35 + label_len * 0.08
    return min(20.0, max(8.0, dynamic))


def format_compact_label(label: str, width: int = 14, max_lines: int = 2) -> str:
    wrapped = textwrap.wrap(
        label.strip(),
        width=width,
        max_lines=max_lines,
        placeholder="...",
        break_long_words=True,
        break_on_hyphens=True,
    )
    return "\n".join(wrapped) if wrapped else "-"


def resolve_label_rotation(labels: list[str]) -> int:
    if not labels:
        return 0
    if len(labels) >= 4:
        return 24
    if max(len(label) for label in labels) > 16:
        return 22
    return 0


def bar_colors(size: int, cmap_name: str = "Blues") -> list[str]:
    if size <= 1:
        return [DEFAULT_BAR_COLOR]
    palette = plt.get_cmap(cmap_name)
    return [palette(0.45 + (index / (size - 1)) * 0.4) for index in range(size)]


def resolve_upper_bound(max_value: float) -> float:
    if max_value <= 0:
        return 1.0
    return max_value * 1.2 + 0.02
