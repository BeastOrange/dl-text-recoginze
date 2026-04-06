from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dltr.data.config import load_data_config
from dltr.data.hardcase import analyze_hardcase_metadata
from dltr.data.inventory import collect_inventories
from dltr.project import ProjectPaths


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

    categories = ["small_text_like", "dense_text_like", "rotated_text_like", "low_quality_like"]
    dataset_names = list(hardcases)
    fig, ax = plt.subplots(figsize=(10, 5))
    for category in categories:
        values = [hardcases[name].keyword_hit_ratio.get(category, 0.0) for name in dataset_names]
        ax.plot(dataset_names, values, marker="o", label=category)
    ax.set_title("Hard-Case Keyword Ratios by Dataset")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
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
