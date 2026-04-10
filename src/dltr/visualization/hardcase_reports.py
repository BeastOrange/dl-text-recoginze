from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dltr.data.config import load_data_config
from dltr.data.hardcase import analyze_hardcase_metadata
from dltr.data.inventory import collect_inventories
from dltr.project import ProjectPaths
from dltr.visualization.plot_style import format_compact_label, resolve_label_rotation, style_axis

HARDCASE_STYLES = [
    ("small_text_like", "Small", "#2F6CAD", "o"),
    ("dense_text_like", "Dense", "#D26D3D", "s"),
    ("rotated_text_like", "Rotated", "#4A8F6D", "^"),
    ("low_quality_like", "Low Quality", "#7E63B8", "D"),
]


def build_hardcase_overview(
    *,
    config_path: Path,
    output_dir: Path,
    project_root: Path,
) -> dict[str, Path]:
    paths = ProjectPaths(
        root=project_root,
        src=project_root / "src",
        configs=project_root / "configs",
        scripts=project_root / "scripts",
        tests=project_root / "tests",
        data_raw=project_root / "data" / "raw",
        data_interim=project_root / "data" / "interim",
        data_processed=project_root / "data" / "processed",
        reports=project_root / "reports",
        artifacts=project_root / "artifacts",
        change_records=project_root / "change_records",
    )
    config = load_data_config(config_path)
    inventories = collect_inventories(paths, config)
    hardcases = {
        name: analyze_hardcase_metadata(inventory)
        for name, inventory in inventories.items()
        if inventory.total_images > 0
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "hardcase_overview.md"
    png_path = output_dir / "hardcase_overview.png"

    dataset_names = list(hardcases)
    fig, ax = plt.subplots(figsize=(_resolve_hardcase_fig_width(dataset_names), 5.2))
    style_axis(ax)
    x_positions = list(range(len(dataset_names)))
    max_ratio = 0.0
    for key, display, color, marker in HARDCASE_STYLES:
        values = [hardcases[name].keyword_hit_ratio.get(key, 0.0) for name in dataset_names]
        if values:
            max_ratio = max(max_ratio, max(values))
        ax.plot(
            x_positions,
            values,
            marker=marker,
            markersize=5.2,
            linewidth=2.0,
            color=color,
            label=display,
        )

    if dataset_names:
        labels = [format_compact_label(name, width=12, max_lines=2) for name in dataset_names]
        rotation = resolve_label_rotation(dataset_names)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
        upper_bound = _resolve_hardcase_upper_bound(max_ratio)
        ax.set_ylim(0.0, upper_bound)
        if max_ratio <= 0.0:
            ax.text(
                0.5,
                0.93,
                "No hard-case keyword hits detected in current metadata.",
                ha="center",
                va="center",
                fontsize=10,
                color="#6B7B8D",
                transform=ax.transAxes,
            )
            for position in x_positions:
                ax.text(
                    position,
                    upper_bound * 0.08,
                    "0.000",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#6B7B8D",
                )
    else:
        ax.set_xticks([])
        ax.set_ylim(0.0, 1.0)
        ax.text(
            0.5,
            0.5,
            "No datasets with images",
            ha="center",
            va="center",
            fontsize=11,
            color="#6B7B8D",
            transform=ax.transAxes,
        )

    ax.set_title("Hard-Case Keyword Ratios by Dataset")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Dataset")
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="#FFFFFF",
        framealpha=0.95,
        edgecolor="#D5DDE5",
    )
    ax.margins(x=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    markdown_path.write_text(
        "\n".join(
            [
                "# Hard-Case Overview",
                "",
                "| Dataset | Images | Small | Dense | Rotated | Low Quality |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            + [
                (
                    f"| {name} | {hardcases[name].total_images} | "
                    f"{hardcases[name].keyword_hit_ratio.get('small_text_like', 0.0):.3f} | "
                    f"{hardcases[name].keyword_hit_ratio.get('dense_text_like', 0.0):.3f} | "
                    f"{hardcases[name].keyword_hit_ratio.get('rotated_text_like', 0.0):.3f} | "
                    f"{hardcases[name].keyword_hit_ratio.get('low_quality_like', 0.0):.3f} |"
                )
                for name in dataset_names
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"markdown": markdown_path, "png": png_path}


def _resolve_hardcase_fig_width(dataset_names: list[str]) -> float:
    if not dataset_names:
        return 9.0
    max_len = max(len(name) for name in dataset_names)
    dynamic = 6.0 + len(dataset_names) * 1.1 + max_len * 0.06
    return min(18.0, max(9.0, dynamic))


def _resolve_hardcase_upper_bound(max_ratio: float) -> float:
    if max_ratio <= 0.0:
        return 0.05
    return max(0.05, max_ratio * 1.2 + 0.02)
