import json

import pytest
from PIL import Image, ImageDraw

from dltr.cli import main

torch = pytest.importorskip("torch")


def test_train_recognizer_returns_nonzero_for_unimplemented_transocr(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    manifest = tmp_path / "data" / "processed" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "val.jsonl"
    charset = tmp_path / "data" / "processed" / "charset.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("", encoding="utf-8")
    val_manifest.write_text("", encoding="utf-8")
    charset.write_text("营\n业\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "recognition" / "transocr.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: transocr_smoke",
                "model_name: transocr",
                "dataset_manifest: data/processed/train.jsonl",
                "validation_manifest: data/processed/val.jsonl",
                "charset_file: data/processed/charset.txt",
                "output_dir: artifacts/checkpoints/recognition/transocr",
                "epochs: 1",
                "batch_size: 2",
                "image_height: 48",
                "image_width: 320",
                "learning_rate: 0.001",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(["train", "recognizer", "--config", str(config_path)])

    assert exit_code == 1


def test_train_recognizer_crnn_runs_smoke(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_text_image(image_a, "营业")
    _write_text_image(image_b, "时间")

    train_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "val.jsonl"
    charset = tmp_path / "data" / "processed" / "charset.txt"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
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
    charset.write_text("营\n业\n时\n间\n", encoding="utf-8")

    config_path = tmp_path / "configs" / "recognition" / "crnn.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: crnn_smoke_cmd",
                "model_name: crnn",
                "dataset_manifest: data/processed/recognition_splits/train.jsonl",
                "validation_manifest: data/processed/recognition_splits/val.jsonl",
                "charset_file: data/processed/charset.txt",
                "output_dir: artifacts/checkpoints/recognition/crnn_smoke_cmd",
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
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "train",
            "recognizer",
            "--config",
            str(config_path),
            "--run-id",
            "cli-smoke",
        ]
    )

    assert exit_code == 0
    assert (
        tmp_path
        / "artifacts"
        / "checkpoints"
        / "recognition"
        / "crnn_smoke_cmd"
        / "cli-smoke"
        / "last.pt"
    ).exists()


def _write_text_image(path, text: str) -> None:
    image = Image.new("L", (128, 32), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), text, fill=0)
    image.save(path)
