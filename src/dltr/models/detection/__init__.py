"""Detection experiment scaffold for DBNet-style workflows."""

from dltr.models.detection.dataset import DetectionInstance, DetectionSample, load_detection_samples
from dltr.models.detection.metrics import compute_detection_scores
from dltr.models.detection.scaffold import (
    DetectionRunConfig,
    DetectionRunContext,
    build_export_plan,
    load_detection_run_config,
    prepare_detection_run,
    write_evaluation_summary,
    write_experiment_metadata,
)
from dltr.models.detection.trainer import DetectionTrainingResult, train_dbnet_detector

__all__ = [
    "DetectionInstance",
    "DetectionRunConfig",
    "DetectionRunContext",
    "DetectionSample",
    "DetectionTrainingResult",
    "build_export_plan",
    "compute_detection_scores",
    "load_detection_samples",
    "load_detection_run_config",
    "prepare_detection_run",
    "train_dbnet_detector",
    "write_evaluation_summary",
    "write_experiment_metadata",
]
