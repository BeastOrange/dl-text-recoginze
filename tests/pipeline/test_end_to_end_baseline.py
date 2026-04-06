from pathlib import Path

from dltr.pipeline.end_to_end import EndToEndLineResult
from dltr.pipeline.end_to_end_baseline import (
    EndToEndBaselineImageResult,
    aggregate_end_to_end_baseline,
    match_predictions_to_ground_truth,
)
from dltr.post_ocr.slots import extract_post_ocr_slots


def test_match_predictions_to_ground_truth_uses_polygon_iou() -> None:
    predictions = [
        EndToEndLineResult(
            line_id="line-0",
            polygon=[10, 10, 110, 10, 110, 40, 10, 40],
            text="营业时间",
            recognition_confidence=0.95,
            analysis_label="service_info",
            analysis_confidence=0.8,
            slots=extract_post_ocr_slots("营业时间"),
        )
    ]
    ground_truth = [
        {"points": [12, 12, 112, 12, 112, 42, 12, 42], "text": "营业时间", "ignore": 0},
        {"points": [200, 200, 260, 200, 260, 240, 200, 240], "text": "特价", "ignore": 0},
    ]

    result = match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold=0.5)

    assert len(result.matches) == 1
    assert result.matches[0].gt_text == "营业时间"
    assert result.total_gt == 2


def test_aggregate_end_to_end_baseline_computes_system_metrics() -> None:
    image_results = [
        EndToEndBaselineImageResult(
            image_path=Path("a.png"),
            total_gt=2,
            matched_lines=2,
            exact_match_lines=1,
            prediction_texts=["营业时间", "特价套餐"],
            target_texts=["营业时间", "特价"],
        ),
        EndToEndBaselineImageResult(
            image_path=Path("b.png"),
            total_gt=1,
            matched_lines=0,
            exact_match_lines=0,
            prediction_texts=[],
            target_texts=[],
        ),
    ]

    summary = aggregate_end_to_end_baseline(image_results)

    assert summary.total_gt_lines == 3
    assert summary.matched_lines == 2
    assert summary.exact_match_lines == 1
    assert summary.detection_coverage == 2 / 3
    assert summary.system_line_accuracy == 1 / 3
