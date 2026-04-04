import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from dltr.models.detection.scaffold import load_detection_run_config
from dltr.models.detection.trainer import train_dbnet_detector
from dltr.project import ProjectPaths

torch = pytest.importorskip("torch")


def test_train_dbnet_detector_runs_smoke_epoch(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset_dir = tmp_path / "data" / "raw" / "rects"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    image_a = tmp_path / "det_a.png"
    image_b = tmp_path / "det_b.png"
    _write_box_image(image_a)
    _write_box_image(image_b)

    train_manifest = tmp_path / "data" / "processed" / "detection_splits" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "detection_splits" / "val.jsonl"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "dataset": "rects",
            "image_path": str(image_a),
            "label_path": str(image_a.with_suffix(".json")),
            "instances": [{"points": [4, 4, 24, 4, 24, 24, 4, 24], "text": "营业", "ignore": 0}],
        },
        {
            "dataset": "shopsign",
            "image_path": str(image_b),
            "label_path": str(image_b.with_suffix(".txt")),
            "instances": [{"points": [6, 6, 26, 6, 26, 26, 6, 26], "text": "时间", "ignore": 0}],
        },
    ]
    train_manifest.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
        encoding="utf-8",
    )
    val_manifest.write_text(train_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    config_path = tmp_path / "configs" / "detection" / "dbnet.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: det_smoke",
                "model_name: dbnet",
                "dataset_dir: data/raw/rects",
                "train_manifest: data/processed/detection_splits/train.jsonl",
                "validation_manifest: data/processed/detection_splits/val.jsonl",
                "output_root: artifacts/detection/det_smoke",
                "epochs: 1",
                "batch_size: 2",
                "learning_rate: 0.001",
                "image_height: 32",
                "image_width: 32",
                "device: cpu",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_detection_run_config(config_path)
    result = train_dbnet_detector(
        config,
        paths=ProjectPaths.from_root(tmp_path),
        run_id="smoke-run",
    )

    assert result.checkpoint_path.exists()
    assert result.best_checkpoint_path.exists()
    assert result.history_path.exists()
    assert result.history_markdown_path.exists()
    assert result.summary_path.exists()
    assert result.report_paths["markdown"].exists()
    history_lines = result.history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1
    assert "train_loss" in history_lines[0]


def _write_box_image(path: Path) -> None:
    image = Image.new("RGB", (32, 32), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 24, 24), outline="black", width=2)
    image.save(path)
