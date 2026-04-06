from pathlib import Path

import pytest
import yaml

from dltr.models.recognition.config import load_recognition_config


def test_load_recognition_config_valid(tmp_path: Path) -> None:
    payload = {
        "experiment_name": "transformer_demo",
        "model_name": "transformer",
        "dataset_manifest": "data/processed/manifest.jsonl",
        "validation_manifest": "data/processed/val.jsonl",
        "charset_file": "data/processed/charset.txt",
        "output_dir": "artifacts/checkpoints/recognition/transformer_demo",
        "epochs": 2,
        "batch_size": 16,
        "image_height": 48,
        "image_width": 320,
        "learning_rate": 0.001,
        "second_pass": {
            "enabled": True,
            "confidence_threshold": 0.8,
            "max_blur_score": 0.4,
            "min_contrast_score": 0.3,
            "min_text_length": 2,
            "min_aspect_ratio": 0.2,
            "max_aspect_ratio": 20.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_recognition_config(config_path)

    assert config.model_name == "transformer"
    assert config.validation_manifest == "data/processed/val.jsonl"
    assert config.second_pass.enabled is True
    assert config.second_pass.confidence_threshold == pytest.approx(0.8)


def test_load_recognition_config_rejects_invalid_model(tmp_path: Path) -> None:
    payload = {
        "experiment_name": "bad_model",
        "model_name": "legacy_attention",
        "dataset_manifest": "data/processed/manifest.jsonl",
        "validation_manifest": "data/processed/val.jsonl",
        "charset_file": "data/processed/charset.txt",
        "output_dir": "artifacts/checkpoints/recognition/bad",
        "epochs": 10,
        "batch_size": 16,
        "image_height": 48,
        "image_width": 320,
        "learning_rate": 0.001,
    }
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model_name"):
        load_recognition_config(config_path)


def test_load_transformer_4090_config_targets_gpu_throughput() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    config = load_recognition_config(
        repo_root / "configs" / "recognition" / "transformer_4090.yaml"
    )

    assert config.model_name == "transformer"
    assert config.batch_size >= 128
    assert config.num_workers >= 4


def test_load_transformer_detector_crop_4090_config_uses_crop_manifests() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    config = load_recognition_config(
        repo_root / "configs" / "recognition" / "transformer_detector_crop_4090.yaml"
    )

    assert config.model_name == "transformer"
    assert "recognition_crop_splits/train.jsonl" in config.dataset_manifest
    assert "recognition_crop_charset_zh_mixed.txt" in config.charset_file
