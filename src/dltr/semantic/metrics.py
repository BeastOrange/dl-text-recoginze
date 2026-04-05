from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticScoreSummary:
    samples: int
    accuracy: float
    macro_f1: float


def compute_semantic_scores(
    *,
    predictions: list[str],
    targets: list[str],
) -> SemanticScoreSummary:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        raise ValueError("predictions must not be empty")

    samples = len(predictions)
    accuracy = (
        sum(pred == target for pred, target in zip(predictions, targets, strict=True)) / samples
    )
    labels = sorted(set(predictions) | set(targets))
    f1_scores = []
    for label in labels:
        tp = sum(
            pred == label and target == label
            for pred, target in zip(predictions, targets, strict=True)
        )
        fp = sum(
            pred == label and target != label
            for pred, target in zip(predictions, targets, strict=True)
        )
        fn = sum(
            pred != label and target == label
            for pred, target in zip(predictions, targets, strict=True)
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1_scores.append(_safe_div(2 * precision * recall, precision + recall))

    return SemanticScoreSummary(
        samples=samples,
        accuracy=accuracy,
        macro_f1=sum(f1_scores) / max(len(f1_scores), 1),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
