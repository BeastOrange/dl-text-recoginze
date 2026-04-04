import json

from dltr.cli import main


def test_report_summarize_training_command_builds_outputs(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    run_a = tmp_path / "artifacts" / "recognition" / "run_a"
    run_b = tmp_path / "artifacts" / "recognition" / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "training_summary.json").write_text(
        json.dumps({"metrics": {"word_accuracy": 0.55, "cer": 0.45}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_b / "training_summary.json").write_text(
        json.dumps({"metrics": {"word_accuracy": 0.72, "cer": 0.28}}, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    exit_code = main(
        [
            "report",
            "summarize-training",
            "--task-name",
            "recognition",
            "--primary-metric",
            "word_accuracy",
            "--run-dirs",
            str(run_a),
            str(run_b),
            "--output-dir",
            "reports/train",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "reports" / "train" / "recognition_summary.json").exists()


def test_report_build_index_and_ablation_commands_write_outputs(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    train_dir = tmp_path / "reports" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "detection_summary.md").write_text("# detection\n", encoding="utf-8")
    (train_dir / "recognition_summary.md").write_text("# recognition\n", encoding="utf-8")
    (train_dir / "project_training_summary.md").write_text("# project\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    index_exit = main(["report", "build-index", "--train-reports-dir", "reports/train"])
    ablation_exit = main(
        [
            "report",
            "build-ablation-template",
            "--task-name",
            "recognition",
            "--experiments",
            "crnn_baseline",
            "transocr_refine",
            "transocr_refine_hardcase",
            "--output-dir",
            "reports/train",
        ]
    )

    assert index_exit == 0
    assert ablation_exit == 0
    assert (tmp_path / "reports" / "train" / "index.md").exists()
    assert (tmp_path / "reports" / "train" / "recognition_ablation_template.md").exists()
