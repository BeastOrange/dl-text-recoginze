import json

from dltr.cli import main


def test_report_summarize_project_command_builds_project_outputs(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    detection_json = tmp_path / "reports" / "train" / "detection_summary.json"
    recognition_json = tmp_path / "reports" / "train" / "recognition_summary.json"
    detection_json.parent.mkdir(parents=True, exist_ok=True)
    detection_json.write_text(
        json.dumps(
            [
                {
                    "run_name": "det_a",
                    "primary_metric": 0.61,
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
                    "primary_metric": 0.81,
                    "best_checkpoint_path": "/tmp/rec_a.pt",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "report",
            "summarize-project",
            "--detection-summary-json",
            str(detection_json),
            "--recognition-summary-json",
            str(recognition_json),
            "--output-dir",
            "reports/train",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "reports" / "train" / "project_training_summary.md").exists()
