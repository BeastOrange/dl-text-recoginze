"""Recognition scaffolding for CRNN and Transformer OCR experiments."""

from .charset import CharacterVocabulary
from .config import RecognitionExperimentConfig, SecondPassConfig, load_recognition_config
from .dataset import RecognitionSample, load_recognition_samples
from .evaluation import RecognitionMetrics, generate_recognition_evaluation_report
from .metrics import RecognitionScoreSummary, compute_recognition_scores
from .preprocessing import (
    RecognitionPreprocessConfig,
    RecognitionPreprocessMeta,
    prepare_recognition_image,
)
from .refinement import QualitySignals, second_pass_reasons, should_apply_second_pass
from .trainer import RecognitionTrainingResult, train_crnn_recognizer, train_transformer_recognizer

__all__ = [
    "CharacterVocabulary",
    "QualitySignals",
    "RecognitionExperimentConfig",
    "RecognitionMetrics",
    "RecognitionPreprocessConfig",
    "RecognitionPreprocessMeta",
    "RecognitionSample",
    "RecognitionScoreSummary",
    "RecognitionTrainingResult",
    "SecondPassConfig",
    "compute_recognition_scores",
    "generate_recognition_evaluation_report",
    "load_recognition_samples",
    "load_recognition_config",
    "second_pass_reasons",
    "should_apply_second_pass",
    "prepare_recognition_image",
    "train_crnn_recognizer",
    "train_transformer_recognizer",
]
