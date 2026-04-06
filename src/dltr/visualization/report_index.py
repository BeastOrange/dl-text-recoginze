from __future__ import annotations

from pathlib import Path


def build_training_report_index(
    *,
    train_reports_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "index.md"
    markdown_files = sorted(
        path for path in train_reports_dir.glob("*.md") if path.name != markdown_path.name
    )
    core_markdown = [
        path
        for path in markdown_files
        if "semantic" not in path.name and "extension" not in path.name
    ]
    extension_markdown = [path for path in markdown_files if path not in core_markdown]
    markdown_path.write_text(
        "\n".join(
            [
                "# Training Report Index",
                "",
                f"- Directory: `{train_reports_dir}`",
                "",
                "## OCR Mainline",
                "",
            ]
            + [f"- [{path.name}]({path.name})" for path in core_markdown]
            + (
                ["", "## Extension Reports", ""]
                + [f"- [{path.name}]({path.name})" for path in extension_markdown]
                if extension_markdown
                else []
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return markdown_path


def build_ablation_template(
    *,
    output_dir: Path,
    task_name: str,
    experiments: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{task_name}_ablation_template.md"
    markdown_path.write_text(
        "\n".join(
            [
                f"# {task_name.title()} Ablation Report Template",
                "",
                "## Experiment Matrix",
                "",
                "| Experiment | Hypothesis | Expected Impact | Actual Result |",
                "|---|---|---|---|",
            ]
            + [f"| {name} |  |  |  |" for name in experiments]
            + [
                "",
                "## Notes",
                "",
                "- Fill this page after running the corresponding experiments.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return markdown_path
