from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def build_ablation_overview(
    *,
    detection_summary_json: Path,
    recognition_summary_json: Path,
    semantic_summary_json: Path | None,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detection = json.loads(detection_summary_json.read_text(encoding="utf-8"))
    recognition = json.loads(recognition_summary_json.read_text(encoding="utf-8"))
    semantic = (
        json.loads(semantic_summary_json.read_text(encoding="utf-8"))
        if semantic_summary_json
        else []
    )

    markdown_path = output_dir / "ablation_overview.md"
    png_path = output_dir / "ablation_overview.png"

    bars = {
        "detection": detection[0]["primary_metric"] if detection else 0.0,
        "recognition": recognition[0]["primary_metric"] if recognition else 0.0,
    }
    if semantic:
        bars["extension"] = semantic[0]["primary_metric"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(bars.keys()), list(bars.values()))
    ax.set_title("OCR Mainline Primary Metric Overview")
    ax.set_ylabel("Primary Metric")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
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
            + ([f"| extension | {bars['extension']:.6f} |"] if "extension" in bars else [])
        )
        + "\n",
        encoding="utf-8",
    )
    return {"markdown": markdown_path, "png": png_path}
