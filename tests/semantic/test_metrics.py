from dltr.semantic.metrics import compute_semantic_scores


def test_compute_semantic_scores_reports_accuracy_and_macro_f1() -> None:
    summary = compute_semantic_scores(
        predictions=["service_info", "advertisement", "service_info"],
        targets=["service_info", "advertisement", "traffic_or_warning"],
    )

    assert summary.samples == 3
    assert summary.accuracy == 2 / 3
    assert 0.0 <= summary.macro_f1 <= 1.0
