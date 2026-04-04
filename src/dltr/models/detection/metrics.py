from __future__ import annotations

from typing import Any


def compute_detection_scores(
    prediction: Any,
    target: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred_mask = (prediction >= threshold).float()
    target_mask = (target >= threshold).float()

    true_positive = float((pred_mask * target_mask).sum().item())
    false_positive = float((pred_mask * (1.0 - target_mask)).sum().item())
    false_negative = float(((1.0 - pred_mask) * target_mask).sum().item())

    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    hmean = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "hmean": hmean,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
