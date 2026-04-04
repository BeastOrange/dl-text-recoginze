from pathlib import Path

import pytest

from dltr.models.recognition.evaluation import (
    RecognitionMetrics,
    generate_recognition_evaluation_report,
)


def test_generate_recognition_evaluation_report_writes_markdown(tmp_path: Path) -> None:
    metrics = RecognitionMetrics(
        samples=120,
        word_accuracy=0.88,
        cer=0.09,
        ned=0.11,
        mean_edit_distance=0.23,
        latency_ms=9.5,
    )
    report_path = generate_recognition_evaluation_report(
        run_name="transocr_baseline",
        model_name="transocr",
        metrics=metrics,
        output_dir=tmp_path,
        notes="Baseline run for Chinese scene-text recognition.",
    )
    content = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "Recognition Evaluation: transocr_baseline" in content
    assert "Word Accuracy" in content


def test_metrics_validation_rejects_invalid_cer() -> None:
    metrics = RecognitionMetrics(
        samples=50,
        word_accuracy=0.8,
        cer=1.2,
        ned=0.2,
        mean_edit_distance=0.5,
    )
    with pytest.raises(ValueError, match="cer"):
        metrics.validate()
