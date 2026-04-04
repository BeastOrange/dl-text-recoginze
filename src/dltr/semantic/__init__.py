"""Semantic analysis scaffolding for OCR outputs."""

from .classes import SEMANTIC_CLASSES, validate_semantic_class
from .report import SemanticPrediction, generate_semantic_report
from .slots import SemanticSlots, extract_semantic_slots

__all__ = [
    "SEMANTIC_CLASSES",
    "SemanticPrediction",
    "SemanticSlots",
    "extract_semantic_slots",
    "generate_semantic_report",
    "validate_semantic_class",
]
