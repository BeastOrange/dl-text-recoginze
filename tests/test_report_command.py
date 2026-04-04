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
