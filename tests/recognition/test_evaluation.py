import json
from pathlib import Path

import pytest

from dltr.models.recognition.evaluation import (
    RecognitionMetrics,
    generate_recognition_evaluation_report,
    write_recognition_evaluation_bundle,
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
        run_name="transformer_baseline",
        model_name="transformer",
        metrics=metrics,
        output_dir=tmp_path,
        notes="Baseline run for Chinese scene-text recognition.",
    )
    content = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "Recognition Evaluation: transformer_baseline" in content
    assert "Word Accuracy" in content


def test_metrics_validation_rejects_invalid_cer() -> None:
    metrics = RecognitionMetrics(
        samples=50,
        word_accuracy=0.8,
        cer=-0.1,
        ned=0.2,
        mean_edit_distance=0.5,
    )
    with pytest.raises(ValueError, match="cer"):
        metrics.validate()


def test_write_recognition_evaluation_bundle_writes_json_with_benchmark_metadata(
    tmp_path: Path,
) -> None:
    metrics = RecognitionMetrics(
        samples=3000,
        word_accuracy=0.91,
        cer=0.07,
        ned=0.08,
        mean_edit_distance=0.18,
    )

    outputs = write_recognition_evaluation_bundle(
        run_name="transformer_iiit5k",
        model_name="transformer",
        metrics=metrics,
        output_dir=tmp_path,
        benchmark_name="iiit5k",
        benchmark_category="main",
        notes="English benchmark evaluation.",
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert outputs["markdown"].exists()
    assert payload["benchmark_name"] == "iiit5k"
    assert payload["benchmark_category"] == "main"
    assert payload["metrics"]["word_accuracy"] == pytest.approx(0.91)
