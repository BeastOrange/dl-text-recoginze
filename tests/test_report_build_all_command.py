import json

from dltr.cli import main


def test_report_build_all_command_generates_report_suite(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")

    det_run = tmp_path / "artifacts" / "detection" / "det_base" / "20260101-000000"
    rec_run = (
        tmp_path / "artifacts" / "checkpoints" / "recognition" / "rec_base" / "20260102-000000"
    )
    det_run.mkdir(parents=True, exist_ok=True)
    rec_run.mkdir(parents=True, exist_ok=True)

    (det_run / "training_summary.json").write_text(
        json.dumps(
            {
                "metrics": {"hmean": 0.61, "precision": 0.63, "recall": 0.60},
                "best_checkpoint_path": str(det_run / "checkpoints" / "best.pt"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (rec_run / "training_summary.json").write_text(
        json.dumps(
            {
                "metrics": {"word_accuracy": 0.78, "cer": 0.22, "ned": 0.18},
                "best_checkpoint_path": str(rec_run / "best.pt"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    exit_code = main(["report", "build-all", "--output-dir", "reports/train"])

    assert exit_code == 0
    assert (tmp_path / "reports" / "train" / "detection_summary.json").exists()
    assert (tmp_path / "reports" / "train" / "recognition_summary.json").exists()
    assert (tmp_path / "reports" / "train" / "project_training_summary.md").exists()
    assert (tmp_path / "reports" / "train" / "index.md").exists()
    assert (tmp_path / "reports" / "train" / "detection_ablation_template.md").exists()
    assert (tmp_path / "reports" / "train" / "recognition_ablation_template.md").exists()
