import json
from pathlib import Path

from dltr.visualization.project_summary import build_project_training_summary


def test_build_project_training_summary_writes_outputs(tmp_path: Path) -> None:
    detection_json = tmp_path / "detection_summary.json"
    recognition_json = tmp_path / "recognition_summary.json"
    semantic_json = tmp_path / "semantic_summary.json"
    detection_json.write_text(
        json.dumps(
            [
                {
                    "run_name": "det_a",
                    "primary_metric": 0.66,
                    "best_checkpoint_path": "/tmp/det_a.pt",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    recognition_json.write_text(
        json.dumps(
            [
                {
                    "run_name": "rec_a",
                    "primary_metric": 0.77,
                    "best_checkpoint_path": "/tmp/rec_a.pt",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    semantic_json.write_text(
        json.dumps(
            [
                {
                    "run_name": "sem_a",
                    "primary_metric": 0.69,
                    "best_checkpoint_path": "/tmp/sem_a.pt",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    outputs = build_project_training_summary(
        detection_summary_json=detection_json,
        recognition_summary_json=recognition_json,
        semantic_summary_json=semantic_json,
        output_dir=tmp_path / "reports",
    )

    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
