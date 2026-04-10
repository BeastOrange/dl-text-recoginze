import json
from pathlib import Path

from dltr.visualization.training_reports import (
    aggregate_training_runs,
    render_detection_history_plot,
    render_recognition_history_plot,
)


def test_render_recognition_history_plot_writes_png_and_markdown(tmp_path: Path) -> None:
    history_path = tmp_path / "recognition_history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "epoch": 1,
                        "train_loss": 1.2,
                        "val_word_accuracy": 0.4,
                        "val_cer": 0.6,
                        "val_ned": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "epoch": 2,
                        "train_loss": 0.8,
                        "val_word_accuracy": 0.6,
                        "val_cer": 0.4,
                        "val_ned": 0.3,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    paths = render_recognition_history_plot(
        run_name="crnn_baseline",
        history_path=history_path,
        output_dir=tmp_path / "reports",
    )

    assert paths["png"].exists()
    assert paths["markdown"].exists()


def test_render_detection_history_plot_writes_png_and_markdown(tmp_path: Path) -> None:
    history_path = tmp_path / "detection_history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "epoch": 1,
                        "train_loss": 1.0,
                        "val_precision": 0.5,
                        "val_recall": 0.4,
                        "val_hmean": 0.44,
                    }
                ),
                json.dumps(
                    {
                        "epoch": 2,
                        "train_loss": 0.7,
                        "val_precision": 0.7,
                        "val_recall": 0.6,
                        "val_hmean": 0.64,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    paths = render_detection_history_plot(
        run_name="dbnet_baseline",
        history_path=history_path,
        output_dir=tmp_path / "reports",
    )

    assert paths["png"].exists()
    assert paths["markdown"].exists()


def test_aggregate_training_runs_writes_summary_outputs(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "training_summary.json").write_text(
        json.dumps(
            {
                "metrics": {"word_accuracy": 0.6, "cer": 0.4},
                "best_checkpoint_path": "/tmp/a.pt",
            }
        ),
        encoding="utf-8",
    )
    (run_b / "training_summary.json").write_text(
        json.dumps(
            {
                "metrics": {"word_accuracy": 0.75, "cer": 0.25},
                "best_checkpoint_path": "/tmp/b.pt",
            }
        ),
        encoding="utf-8",
    )

    outputs = aggregate_training_runs(
        run_dirs=[run_a, run_b],
        output_dir=tmp_path / "summary",
        task_name="recognition",
        primary_metric="word_accuracy",
    )
    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
    assert outputs["png"].exists()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert [item["run_name"] for item in payload] == ["run_b", "run_a"]


def test_aggregate_training_runs_excludes_obsolete_smoke_runs(tmp_path: Path) -> None:
    smoke_run = tmp_path / "crnn_report_smoke" / "report-smoke"
    real_run = tmp_path / "transformer_baseline_4090" / "20260406-120000"
    smoke_run.mkdir(parents=True)
    real_run.mkdir(parents=True)
    (smoke_run / "training_summary.json").write_text(
        json.dumps(
            {
                "run_id": "report-smoke",
                "metrics": {"word_accuracy": 0.99},
                "best_checkpoint_path": "/tmp/smoke.pt",
            }
        ),
        encoding="utf-8",
    )
    (real_run / "training_summary.json").write_text(
        json.dumps(
            {
                "run_id": "20260406-120000",
                "metrics": {"word_accuracy": 0.78},
                "best_checkpoint_path": "/tmp/real.pt",
            }
        ),
        encoding="utf-8",
    )

    outputs = aggregate_training_runs(
        run_dirs=[smoke_run, real_run],
        output_dir=tmp_path / "summary",
        task_name="recognition",
        primary_metric="word_accuracy",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert [item["run_name"] for item in payload] == ["20260406-120000"]


def test_aggregate_training_runs_handles_long_labels_and_writes_nonempty_png(
    tmp_path: Path,
) -> None:
    long_name = "transformer_detector_crop_cn_scene_4090_multitask_20260406_153844"
    second_name = "recdet_20260406_201309_extremely_long_variant_name"
    run_a = tmp_path / long_name
    run_b = tmp_path / second_name
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "training_summary.json").write_text(
        json.dumps({"metrics": {"word_accuracy": 0.73}, "best_checkpoint_path": "/tmp/a.pt"}),
        encoding="utf-8",
    )
    (run_b / "training_summary.json").write_text(
        json.dumps({"metrics": {"word_accuracy": 0.69}, "best_checkpoint_path": "/tmp/b.pt"}),
        encoding="utf-8",
    )

    outputs = aggregate_training_runs(
        run_dirs=[run_a, run_b],
        output_dir=tmp_path / "summary",
        task_name="recognition",
        primary_metric="word_accuracy",
    )

    assert outputs["png"].exists()
    assert outputs["png"].stat().st_size > 0


def test_aggregate_training_runs_handles_empty_input(tmp_path: Path) -> None:
    outputs = aggregate_training_runs(
        run_dirs=[],
        output_dir=tmp_path / "summary",
        task_name="recognition",
        primary_metric="word_accuracy",
    )

    assert outputs["png"].exists()
    assert outputs["png"].stat().st_size > 0
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload == []
