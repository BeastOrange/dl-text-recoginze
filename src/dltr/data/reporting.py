from __future__ import annotations

from pathlib import Path

from dltr.data.types import DataConfig, DatasetInventory, HardCaseMetadata, ValidationSummary
from dltr.project import ProjectPaths
from dltr.visualization.eda_markdown import render_eda_markdown


def write_eda_markdown_report(
    project_paths: ProjectPaths,
    config: DataConfig,
    validation: ValidationSummary,
    inventories: dict[str, DatasetInventory],
    hardcases: dict[str, HardCaseMetadata],
    filename: str = "dataset_eda_summary.md",
) -> Path:
    report_dir = project_paths.reports / "eda"
    report_dir.mkdir(parents=True, exist_ok=True)
    content = render_eda_markdown(
        config=config,
        validation=validation,
        inventories=inventories,
        hardcases=hardcases,
    )
    output_path = report_dir / filename
    output_path.write_text(content, encoding="utf-8")
    return output_path
