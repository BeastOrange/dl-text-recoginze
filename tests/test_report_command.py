import json

import pytest

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


def test_evaluate_recognizer_command_writes_json_bundle(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "evaluate",
            "recognizer",
            "--run-name",
            "transformer_iiit5k",
            "--model-name",
            "transformer",
            "--samples",
            "3000",
            "--word-accuracy",
            "0.91",
            "--cer",
            "0.07",
            "--ned",
            "0.08",
            "--mean-edit-distance",
            "0.18",
            "--benchmark-name",
            "iiit5k",
            "--benchmark-category",
            "main",
            "--output-dir",
            "reports/eval",
        ]
    )

    assert exit_code == 0
    payload = json.loads(
        (
            tmp_path / "reports" / "eval" / "transformer_iiit5k_recognition_eval.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["benchmark_name"] == "iiit5k"
    assert payload["benchmark_category"] == "main"


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
            "transformer_baseline",
            "transformer_hardcase",
            "--output-dir",
            "reports/train",
        ]
    )

    assert index_exit == 0
    assert ablation_exit == 0
    assert (tmp_path / "reports" / "train" / "index.md").exists()
    assert (tmp_path / "reports" / "train" / "recognition_ablation_template.md").exists()


def test_report_summarize_english_benchmark_command_builds_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    eval_dir = tmp_path / "reports" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for name, score, category in (
        ("iiit5k", 0.91, "main"),
        ("svt", 0.89, "main"),
        ("icdar13", 0.92, "main"),
        ("icdar15", 0.76, "hard"),
    ):
        (eval_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "run_name": f"transformer_{name}",
                    "model_name": "transformer",
                    "benchmark_name": name,
                    "benchmark_category": category,
                    "metrics": {
                        "samples": 100,
                        "word_accuracy": score,
                        "cer": 0.1,
                        "ned": 0.1,
                        "mean_edit_distance": 0.2,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    monkeypatch.chdir(tmp_path)
    exit_code = main(
        [
            "report",
            "summarize-english-benchmark",
            "--benchmark-jsons",
            "reports/eval/iiit5k.json",
            "reports/eval/svt.json",
            "reports/eval/icdar13.json",
            "reports/eval/icdar15.json",
            "--output-dir",
            "reports/english",
        ]
    )

    assert exit_code == 0
    payload = json.loads(
        (tmp_path / "reports" / "english" / "english_benchmark_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["summary"]["main_average_word_accuracy"] == pytest.approx(
        (0.91 + 0.89 + 0.92) / 3
    )
    assert payload["summary"]["hard_average_word_accuracy"] == pytest.approx(0.76)
