from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dltr.semantic.classes import SEMANTIC_CLASSES


@dataclass(frozen=True)
class SemanticExperimentConfig:
    experiment_name: str
    model_name: str
    label_set: list[str]
    dataset_manifest: str
    validation_manifest: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    device: str = "auto"
    num_workers: int = 0

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not self.label_set:
            raise ValueError("label_set must not be empty")
        invalid = sorted({label for label in self.label_set if label not in SEMANTIC_CLASSES})
        if invalid:
            raise ValueError(f"Unsupported semantic labels: {', '.join(invalid)}")
        if not self.dataset_manifest.strip():
            raise ValueError("dataset_manifest must be non-empty")
        if not self.validation_manifest.strip():
            raise ValueError("validation_manifest must be non-empty")
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SemanticExperimentConfig:
        config = cls(
            experiment_name=str(payload.get("experiment_name", "")).strip(),
            model_name=str(payload.get("model_name", "")).strip(),
            label_set=[str(item).strip() for item in payload.get("label_set", [])],
            dataset_manifest=str(payload.get("dataset_manifest", "")).strip(),
            validation_manifest=str(payload.get("validation_manifest", "")).strip(),
            output_dir=str(payload.get("output_dir", "")).strip(),
            epochs=int(payload.get("epochs", 0)),
            batch_size=int(payload.get("batch_size", 0)),
            learning_rate=float(payload.get("learning_rate", 0.0)),
            max_length=int(payload.get("max_length", 0)),
            device=str(payload.get("device", "auto")).strip() or "auto",
            num_workers=int(payload.get("num_workers", 0)),
        )
        config.validate()
        return config


def load_semantic_config(config_path: str | Path) -> SemanticExperimentConfig:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Semantic config must be a YAML mapping")
    return SemanticExperimentConfig.from_dict(payload)
