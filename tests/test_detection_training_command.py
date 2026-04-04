import json

import pytest
from PIL import Image, ImageDraw

from dltr.cli import main

torch = pytest.importorskip("torch")


def test_train_detector_runs_smoke(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    (tmp_path / "data" / "raw" / "rects").mkdir(parents=True, exist_ok=True)
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_box_image(image_a)
    _write_box_image(image_b)

    train_manifest = tmp_path / "data" / "processed" / "detection_splits" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "detection_splits" / "val.jsonl"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    train_manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(image_a),
                        "label_path": str(image_a.with_suffix(".json")),
                        "instances": [
                            {
                                "points": [4, 4, 24, 4, 24, 24, 4, 24],
                                "text": "营业",
                                "ignore": 0,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "shopsign",
                        "image_path": str(image_b),
                        "label_path": str(image_b.with_suffix(".txt")),
                        "instances": [
                            {
                                "points": [6, 6, 26, 6, 26, 26, 6, 26],
                                "text": "时间",
                                "ignore": 0,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    val_manifest.write_text(train_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    config_path = tmp_path / "configs" / "detection" / "dbnet.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: det_cmd_smoke",
                "model_name: dbnet",
                "dataset_dir: data/raw/rects",
                "train_manifest: data/processed/detection_splits/train.jsonl",
                "validation_manifest: data/processed/detection_splits/val.jsonl",
                "output_root: artifacts/detection/det_cmd_smoke",
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
    monkeypatch.chdir(tmp_path)

    exit_code = main(["train", "detector", "--config", str(config_path), "--run-id", "cli-smoke"])

    assert exit_code == 0
    assert (
        tmp_path
        / "artifacts"
        / "detection"
        / "det_cmd_smoke"
        / "cli-smoke"
        / "checkpoints"
        / "best.pt"
    ).exists()


def _write_box_image(path) -> None:
    image = Image.new("RGB", (32, 32), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 24, 24), outline="black", width=2)
    image.save(path)
