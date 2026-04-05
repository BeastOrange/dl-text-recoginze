"""Semantic analysis scaffolding for OCR outputs."""

from .classes import SEMANTIC_CLASSES, validate_semantic_class
from .classification import SemanticClassification, classify_scene_text
from .config import SemanticExperimentConfig, load_semantic_config
from .dataset import SemanticSample, load_semantic_samples
from .metrics import SemanticScoreSummary, compute_semantic_scores
from .report import SemanticPrediction, generate_semantic_report
from .slots import SemanticSlots, extract_semantic_slots
from .trainer import SemanticTrainingResult, train_semantic_classifier

__all__ = [
    "SEMANTIC_CLASSES",
    "SemanticClassification",
    "SemanticExperimentConfig",
    "SemanticPrediction",
    "SemanticSample",
    "SemanticScoreSummary",
    "SemanticTrainingResult",
    "SemanticSlots",
    "classify_scene_text",
    "compute_semantic_scores",
    "extract_semantic_slots",
    "generate_semantic_report",
    "load_semantic_config",
    "load_semantic_samples",
    "train_semantic_classifier",
    "validate_semantic_class",
]
