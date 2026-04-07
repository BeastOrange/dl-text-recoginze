from __future__ import annotations

import json
from pathlib import Path

MAINLINE_TASKS = ("detection", "recognition")


def build_project_training_summary(
    *,
    detection_summary_json: Path,
    recognition_summary_json: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detection = json.loads(detection_summary_json.read_text(encoding="utf-8"))
    recognition = json.loads(recognition_summary_json.read_text(encoding="utf-8"))

    payload = {
        "detection": detection,
        "recognition": recognition,
    }
    json_path = output_dir / "project_training_summary.json"
    markdown_path = output_dir / "project_training_summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# Project Training Summary",
                "",
                "> Mainline OCR summary includes detection and recognition only.",
                "> Obsolete smoke and extension artifacts are excluded.",
                "",
                "## Detection",
                "",
            ]
            + _format_task_lines(detection)
            + [
                "",
                "## Recognition",
                "",
            ]
            + _format_task_lines(recognition)
        )
        + "\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}

def _format_task_lines(records: list[dict[str, object]]) -> list[str]:
    if not records:
        return ["- No mainline runs available."]
    return [f"- `{item['run_name']}`: `{item['primary_metric']:.6f}`" for item in records]
