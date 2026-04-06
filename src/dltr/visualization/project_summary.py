from __future__ import annotations

import json
from pathlib import Path


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
                "## Detection",
                "",
            ]
            + [
                f"- `{item['run_name']}`: `{item['primary_metric']:.6f}`"
                for item in detection
            ]
            + [
                "",
                "## Recognition",
                "",
            ]
            + [
                f"- `{item['run_name']}`: `{item['primary_metric']:.6f}`"
                for item in recognition
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}
