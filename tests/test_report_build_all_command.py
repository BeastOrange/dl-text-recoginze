import json

from dltr.cli import main


def test_report_build_all_command_generates_report_suite(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")

    smoke_det_run = (
        tmp_path / "artifacts" / "detection" / "det_dbnet_report_smoke" / "report-smoke"
    )
    real_det_run = tmp_path / "artifacts" / "detection" / "det_base" / "20260101-000000"
    smoke_rec_run = (
        tmp_path
        / "artifacts"
        / "checkpoints"
        / "recognition"
        / "crnn_report_smoke"
        / "report-smoke"
    )
    real_rec_run = (
        tmp_path / "artifacts" / "checkpoints" / "recognition" / "rec_base" / "20260102-000000"
    )
    for path in (smoke_det_run, real_det_run, smoke_rec_run, real_rec_run):
        path.mkdir(parents=True, exist_ok=True)

    (smoke_det_run / "training_summary.json").write_text(
        json.dumps(
            {
                "run_id": "report-smoke",
                "metrics": {"hmean": 0.99, "precision": 0.99, "recall": 0.99},
                "best_checkpoint_path": str(smoke_det_run / "checkpoints" / "best.pt"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (real_det_run / "training_summary.json").write_text(
        json.dumps(
            {
                "metrics": {"hmean": 0.61, "precision": 0.63, "recall": 0.60},
                "best_checkpoint_path": str(real_det_run / "checkpoints" / "best.pt"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (smoke_rec_run / "training_summary.json").write_text(
        json.dumps(
            {
                "run_id": "report-smoke",
                "metrics": {"word_accuracy": 0.99, "cer": 0.01, "ned": 0.01},
                "best_checkpoint_path": str(smoke_rec_run / "best.pt"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (real_rec_run / "training_summary.json").write_text(
        json.dumps(
            {
                "metrics": {"word_accuracy": 0.78, "cer": 0.22, "ned": 0.18},
                "best_checkpoint_path": str(real_rec_run / "best.pt"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    exit_code = main(["report", "build-all", "--output-dir", "reports/train"])

    assert exit_code == 0
    detection_summary = json.loads(
        (tmp_path / "reports" / "train" / "detection_summary.json").read_text(encoding="utf-8")
    )
    recognition_summary = json.loads(
        (tmp_path / "reports" / "train" / "recognition_summary.json").read_text(encoding="utf-8")
    )
    project_summary = (tmp_path / "reports" / "train" / "project_training_summary.md").read_text(
        encoding="utf-8"
    )

    assert [item["run_name"] for item in detection_summary] == ["20260101-000000"]
    assert [item["run_name"] for item in recognition_summary] == ["20260102-000000"]
    assert "report-smoke" not in project_summary
    assert (tmp_path / "reports" / "train" / "index.md").exists()
    assert (tmp_path / "reports" / "train" / "detection_ablation_template.md").exists()
    assert (tmp_path / "reports" / "train" / "recognition_ablation_template.md").exists()
