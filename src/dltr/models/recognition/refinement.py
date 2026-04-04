from __future__ import annotations

from dataclasses import dataclass

from .config import SecondPassConfig


@dataclass(frozen=True)
class QualitySignals:
    blur_score: float | None = None
    contrast_score: float | None = None
    aspect_ratio: float | None = None


def second_pass_reasons(
    confidence: float,
    text: str,
    quality: QualitySignals,
    policy: SecondPassConfig,
) -> list[str]:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    reasons: list[str] = []
    text_length = len(text.strip())
    if confidence < policy.confidence_threshold:
        reasons.append("low_confidence")
    if text_length < policy.min_text_length:
        reasons.append("short_text")
    if quality.blur_score is not None and quality.blur_score > policy.max_blur_score:
        reasons.append("high_blur")
    if quality.contrast_score is not None and quality.contrast_score < policy.min_contrast_score:
        reasons.append("low_contrast")
    if quality.aspect_ratio is not None:
        if (
            quality.aspect_ratio < policy.min_aspect_ratio
            or quality.aspect_ratio > policy.max_aspect_ratio
        ):
            reasons.append("extreme_aspect_ratio")
    return reasons


def should_apply_second_pass(
    confidence: float,
    text: str,
    quality: QualitySignals,
    policy: SecondPassConfig,
) -> bool:
    if not policy.enabled:
        return False
    reasons = second_pass_reasons(
        confidence=confidence,
        text=text,
        quality=quality,
        policy=policy,
    )
    return len(reasons) > 0
