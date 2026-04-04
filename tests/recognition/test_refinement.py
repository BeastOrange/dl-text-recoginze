from dltr.models.recognition.config import SecondPassConfig
from dltr.models.recognition.refinement import (
    QualitySignals,
    second_pass_reasons,
    should_apply_second_pass,
)


def test_second_pass_reasons_for_low_confidence() -> None:
    policy = SecondPassConfig(enabled=True, confidence_threshold=0.85)
    reasons = second_pass_reasons(
        confidence=0.5,
        text="测试文本",
        quality=QualitySignals(blur_score=0.2, contrast_score=0.8, aspect_ratio=2.0),
        policy=policy,
    )
    assert "low_confidence" in reasons


def test_second_pass_triggers_for_quality_issues() -> None:
    policy = SecondPassConfig(enabled=True, max_blur_score=0.3, min_contrast_score=0.4)
    trigger = should_apply_second_pass(
        confidence=0.95,
        text="高质量文本",
        quality=QualitySignals(blur_score=0.7, contrast_score=0.2, aspect_ratio=2.0),
        policy=policy,
    )
    assert trigger is True


def test_second_pass_not_triggered_for_good_case() -> None:
    policy = SecondPassConfig(enabled=True, confidence_threshold=0.7)
    trigger = should_apply_second_pass(
        confidence=0.98,
        text="欢迎光临",
        quality=QualitySignals(blur_score=0.1, contrast_score=0.9, aspect_ratio=3.0),
        policy=policy,
    )
    assert trigger is False
