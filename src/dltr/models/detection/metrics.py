from __future__ import annotations

from typing import Any

import numpy as np


def compute_detection_scores(
    prediction: Any,
    target: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred_mask = (_to_numpy_mask(prediction) >= threshold).astype(np.float32)
    target_mask = (_to_numpy_mask(target) >= threshold).astype(np.float32)

    true_positive = float((pred_mask * target_mask).sum())
    false_positive = float((pred_mask * (1.0 - target_mask)).sum())
    false_negative = float(((1.0 - pred_mask) * target_mask).sum())

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


def _to_numpy_mask(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)
