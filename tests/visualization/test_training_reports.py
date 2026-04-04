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
