"""Recognition scaffolding for CRNN/TransOCR experiments."""

from .config import RecognitionExperimentConfig, SecondPassConfig, load_recognition_config
from .evaluation import RecognitionMetrics, generate_recognition_evaluation_report
from .refinement import QualitySignals, second_pass_reasons, should_apply_second_pass

__all__ = [
    "QualitySignals",
    "RecognitionExperimentConfig",
    "RecognitionMetrics",
    "SecondPassConfig",
    "generate_recognition_evaluation_report",
    "load_recognition_config",
    "second_pass_reasons",
    "should_apply_second_pass",
]
