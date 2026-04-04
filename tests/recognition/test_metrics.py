from dltr.models.recognition.metrics import (
    RecognitionScoreSummary,
    compute_recognition_scores,
)


def test_compute_recognition_scores_reports_accuracy_and_cer() -> None:
    summary = compute_recognition_scores(
        predictions=["营业时间", "电话123"],
        targets=["营业时间", "电话124"],
    )

    assert isinstance(summary, RecognitionScoreSummary)
    assert summary.samples == 2
    assert summary.word_accuracy == 0.5
    assert summary.cer > 0.0
    assert summary.ned >= 0.0
