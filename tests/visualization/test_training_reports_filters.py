from __future__ import annotations

import json
from pathlib import Path

from dltr.visualization.training_reports import aggregate_training_runs


def test_aggregate_training_runs_keeps_non_smoke_semantic_extension_runs(tmp_path: Path) -> None:
    semantic_extension_run = tmp_path / "semantic_extension_real" / "20260407-000000"
    semantic_extension_run.mkdir(parents=True)
    (semantic_extension_run / "training_summary.json").write_text(
        json.dumps(
            {
                "run_id": "20260407-000000",
                "metrics": {"accuracy": 0.55},
                "best_checkpoint_path": "/tmp/extension.pt",
            }
        ),
        encoding="utf-8",
    )

    outputs = aggregate_training_runs(
        run_dirs=[semantic_extension_run],
        output_dir=tmp_path / "summary",
        task_name="extension",
        primary_metric="accuracy",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert [item["run_name"] for item in payload] == ["20260407-000000"]
