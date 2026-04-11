from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .preprocessing import RecognitionPreprocessConfig

SUPPORTED_MODELS = {"crnn", "transformer"}


@dataclass(frozen=True)
class SecondPassConfig:
    enabled: bool = True
    confidence_threshold: float = 0.78
    max_blur_score: float = 0.45
    min_contrast_score: float = 0.35
    min_text_length: int = 2
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 20.0

    def validate(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("second_pass.confidence_threshold must be in [0, 1]")
        if not 0.0 <= self.max_blur_score <= 1.0:
            raise ValueError("second_pass.max_blur_score must be in [0, 1]")
        if not 0.0 <= self.min_contrast_score <= 1.0:
            raise ValueError("second_pass.min_contrast_score must be in [0, 1]")
        if self.min_text_length < 1:
            raise ValueError("second_pass.min_text_length must be >= 1")
        if self.min_aspect_ratio <= 0:
            raise ValueError("second_pass.min_aspect_ratio must be > 0")
        if self.max_aspect_ratio < self.min_aspect_ratio:
            raise ValueError("second_pass.max_aspect_ratio must be >= min_aspect_ratio")


@dataclass(frozen=True)
class RecognitionExperimentConfig:
    experiment_name: str
    model_name: str
    dataset_manifest: str
    validation_manifest: str
    charset_file: str
    output_dir: str
    epochs: int
    batch_size: int
    image_height: int
    image_width: int
    learning_rate: float
    monitor_metric: str = "word_accuracy"
    lr_scheduler_patience: int | None = None
    lr_scheduler_factor: float = 0.5
    min_learning_rate: float = 1e-5
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    second_pass: SecondPassConfig = field(default_factory=SecondPassConfig)
    preprocess: RecognitionPreprocessConfig | None = None
    device: str = "auto"
    num_workers: int = 0

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if self.model_name.lower() not in SUPPORTED_MODELS:
            raise ValueError(f"model_name must be one of {sorted(SUPPORTED_MODELS)}")
        if not self.dataset_manifest.strip():
            raise ValueError("dataset_manifest must be non-empty")
        if not self.validation_manifest.strip():
            raise ValueError("validation_manifest must be non-empty")
        if not self.charset_file.strip():
            raise ValueError("charset_file must be non-empty")
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.monitor_metric not in {"word_accuracy", "cer", "ned"}:
            raise ValueError("monitor_metric must be one of {'word_accuracy', 'cer', 'ned'}")
        if self.lr_scheduler_patience is not None and self.lr_scheduler_patience < 0:
            raise ValueError("lr_scheduler_patience must be >= 0 when provided")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("lr_scheduler_factor must be in (0, 1)")
        if self.min_learning_rate <= 0:
            raise ValueError("min_learning_rate must be > 0")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0 when provided")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        self.second_pass.validate()
        if self.preprocess is None:
            raise ValueError("preprocess must be configured")
        self.preprocess.validate()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RecognitionExperimentConfig:
        second_pass_raw = payload.get("second_pass", {})
        preprocess_raw = payload.get("preprocess", {})
        image_height = int(payload.get("image_height", 0))
        image_width = int(payload.get("image_width", 0))
        second_pass = SecondPassConfig(
            enabled=bool(second_pass_raw.get("enabled", True)),
            confidence_threshold=float(second_pass_raw.get("confidence_threshold", 0.78)),
            max_blur_score=float(second_pass_raw.get("max_blur_score", 0.45)),
            min_contrast_score=float(second_pass_raw.get("min_contrast_score", 0.35)),
            min_text_length=int(second_pass_raw.get("min_text_length", 2)),
            min_aspect_ratio=float(second_pass_raw.get("min_aspect_ratio", 0.2)),
            max_aspect_ratio=float(second_pass_raw.get("max_aspect_ratio", 20.0)),
        )
        preprocess = RecognitionPreprocessConfig(
            target_height=image_height,
            target_width=image_width,
            preserve_aspect_ratio=bool(preprocess_raw.get("preserve_aspect_ratio", True)),
            rotate_vertical_text=bool(preprocess_raw.get("rotate_vertical_text", True)),
            vertical_aspect_threshold=float(
                preprocess_raw.get("vertical_aspect_threshold", 1.2)
            ),
            padding_value=int(preprocess_raw.get("padding_value", 255)),
        )
        config = cls(
            experiment_name=str(payload.get("experiment_name", "")).strip(),
            model_name=str(payload.get("model_name", "")).strip().lower(),
            dataset_manifest=str(payload.get("dataset_manifest", "")).strip(),
            validation_manifest=str(payload.get("validation_manifest", "")).strip(),
            charset_file=str(payload.get("charset_file", "")).strip(),
            output_dir=str(payload.get("output_dir", "")).strip(),
            epochs=int(payload.get("epochs", 0)),
            batch_size=int(payload.get("batch_size", 0)),
            image_height=image_height,
            image_width=image_width,
            learning_rate=float(payload.get("learning_rate", 0.0)),
            monitor_metric=str(payload.get("monitor_metric", "word_accuracy")).strip(),
            lr_scheduler_patience=(
                int(payload["lr_scheduler_patience"])
                if payload.get("lr_scheduler_patience") is not None
                else None
            ),
            lr_scheduler_factor=float(payload.get("lr_scheduler_factor", 0.5)),
            min_learning_rate=float(payload.get("min_learning_rate", 1e-5)),
            early_stopping_patience=(
                int(payload["early_stopping_patience"])
                if payload.get("early_stopping_patience") is not None
                else None
            ),
            early_stopping_min_delta=float(payload.get("early_stopping_min_delta", 0.0)),
            device=str(payload.get("device", "auto")).strip() or "auto",
            num_workers=int(payload.get("num_workers", 0)),
            second_pass=second_pass,
            preprocess=preprocess,
        )
        config.validate()
        return config


def load_recognition_config(config_path: str | Path) -> RecognitionExperimentConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Recognition config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Recognition config must be a YAML mapping")
    return RecognitionExperimentConfig.from_dict(payload)
