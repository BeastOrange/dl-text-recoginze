import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from dltr.models.recognition.config import load_recognition_config
from dltr.models.recognition.trainer import train_crnn_recognizer
from dltr.project import ProjectPaths

torch = pytest.importorskip("torch")


def test_train_crnn_recognizer_runs_smoke_epoch(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")

    train_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "val.jsonl"
    charset_path = tmp_path / "data" / "processed" / "charset_zh_mixed.txt"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    charset_path.parent.mkdir(parents=True, exist_ok=True)

    image_a = tmp_path / "sample_a.png"
    image_b = tmp_path / "sample_b.png"
    _write_text_image(image_a, "营业")
    _write_text_image(image_b, "时间")
    train_manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(image_a),
                        "label_path": str(image_a.with_suffix(".json")),
                        "text": "营业",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "shopsign",
                        "image_path": str(image_b),
                        "label_path": str(image_b.with_suffix(".txt")),
                        "text": "时间",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    val_manifest.write_text(train_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    charset_path.write_text("营\n业\n时\n间\n", encoding="utf-8")

    config_path = tmp_path / "configs" / "recognition" / "crnn.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: crnn_smoke",
                "model_name: crnn",
                "dataset_manifest: data/processed/recognition_splits/train.jsonl",
                "validation_manifest: data/processed/recognition_splits/val.jsonl",
                "charset_file: data/processed/charset_zh_mixed.txt",
                "output_dir: artifacts/checkpoints/recognition/crnn_smoke",
                "epochs: 1",
                "batch_size: 2",
                "image_height: 32",
                "image_width: 128",
                "learning_rate: 0.001",
                "device: cpu",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_recognition_config(config_path)
    result = train_crnn_recognizer(
        config,
        paths=ProjectPaths.from_root(tmp_path),
        run_id="smoke-run",
    )

    assert result.checkpoint_path.exists()
    assert result.best_checkpoint_path.exists()
    assert result.history_path.exists()
    assert result.history_markdown_path.exists()
    assert result.history_plot_path.exists()
    assert result.summary_path.exists()
    assert result.report_path.exists()
    assert result.metrics.samples == 2
    history_lines = result.history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1
    assert "train_loss" in history_lines[0]


def _write_text_image(path: Path, text: str) -> None:
    image = Image.new("L", (128, 32), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), text, fill=0)
    image.save(path)
