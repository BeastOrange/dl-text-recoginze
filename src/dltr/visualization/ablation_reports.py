from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from dltr.visualization.plot_style import resolve_upper_bound, style_axis


def build_ablation_overview(
    *,
    detection_summary_json: Path,
    recognition_summary_json: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detection = json.loads(detection_summary_json.read_text(encoding="utf-8"))
    recognition = json.loads(recognition_summary_json.read_text(encoding="utf-8"))

    markdown_path = output_dir / "ablation_overview.md"
    png_path = output_dir / "ablation_overview.png"

    bars = {
        "detection": detection[0]["primary_metric"] if detection else 0.0,
        "recognition": recognition[0]["primary_metric"] if recognition else 0.0,
    }
    labels = list(bars.keys())
    values = list(bars.values())
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    style_axis(ax)
    rendered = ax.bar(
        labels,
        values,
        color=["#2F6CAD", "#4A8F6D"],
        edgecolor="#2E3A46",
        linewidth=0.6,
        width=0.62,
    )
    upper_bound = resolve_upper_bound(max(values))
    ax.set_ylim(0.0, upper_bound)
    for bar, value in zip(rendered, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper_bound * 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2E3A46",
        )
    ax.set_title("OCR Mainline Primary Metric Overview")
    ax.set_ylabel("Primary Metric")
    ax.set_xlabel("Task")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    markdown_path.write_text(
        "\n".join(
            [
                "# Ablation Overview",
                "",
                "> Detection and recognition are the OCR mainline tasks.",
                "> Extension modules are listed separately and do not define",
                "> the mainline conclusion.",
                "",
                "| Task | Primary Metric |",
                "|---|---:|",
                f"| detection | {bars['detection']:.6f} |",
                f"| recognition | {bars['recognition']:.6f} |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"markdown": markdown_path, "png": png_path}
