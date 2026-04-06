from pathlib import Path

from dltr.pipeline.end_to_end import EndToEndLineResult
from dltr.pipeline.end_to_end_baseline import (
    EndToEndBaselineImageResult,
    aggregate_end_to_end_baseline,
    evaluate_end_to_end_manifest,
    match_predictions_to_ground_truth,
    sweep_end_to_end_manifest,
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


def test_evaluate_end_to_end_manifest_updates_progress(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "val.jsonl"
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"fake")
    image_b.write_bytes(b"fake")
    manifest_path.write_text(
        "\n".join(
            [
                (
                    '{"image_path": "'
                    + str(image_a)
                    + '", "instances": [{"points": [10, 10, 110, 10, 110, 40, 10, 40], '
                    '"text": "营业时间", "ignore": 0}]}'
                ),
                (
                    '{"image_path": "'
                    + str(image_b)
                    + '", "instances": [{"points": [20, 20, 100, 20, 100, 60, 20, 60], '
                    '"text": "特价", "ignore": 0}]}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    progress_calls: list[tuple[str, int, dict[str, float | int]]] = []

    class _FakeProgressBar:
        def __init__(self, total: int, description: str) -> None:
            progress_calls.append(("init", total, {"description": description}))

        def update(self, current: int, *, metrics=None) -> None:  # noqa: ANN001
            progress_calls.append(("update", current, metrics or {}))

        def finish(self, *, metrics=None) -> None:  # noqa: ANN001
            progress_calls.append(("finish", -1, metrics or {}))

    class _DetectorSession:
        @classmethod
        def from_checkpoint(cls, checkpoint_path):  # noqa: ANN001
            return cls()

    class _RecognizerSession:
        @classmethod
        def from_checkpoint(cls, checkpoint_path):  # noqa: ANN001
            return cls()

    def fake_infer_end_to_end_image(**kwargs):  # noqa: ANN001
        image_path = kwargs["image_path"]
        if image_path == image_a:
            line_results = [
                EndToEndLineResult(
                    line_id="line-0",
                    polygon=[10, 10, 110, 10, 110, 40, 10, 40],
                    text="营业时间",
                    recognition_confidence=0.95,
                    analysis_label="service_info",
                    analysis_confidence=0.9,
                    slots=extract_post_ocr_slots("营业时间"),
                )
            ]
        else:
            line_results = []
        return None, line_results

    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline.ProgressBar",
        _FakeProgressBar,
    )
    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline.DetectionPredictorSession",
        _DetectorSession,
    )
    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline.RecognitionPredictorSession",
        _RecognizerSession,
    )
    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline.infer_end_to_end_image",
        fake_infer_end_to_end_image,
    )

    evaluate_end_to_end_manifest(
        manifest_path=manifest_path,
        output_dir=tmp_path / "reports" / "eval",
        detector_checkpoint=tmp_path / "det.pt",
        recognizer_checkpoint=tmp_path / "rec.pt",
        max_images=None,
    )

    assert progress_calls[0] == ("init", 2, {"description": "系统评估"})
    assert progress_calls[1][0] == "update"
    assert progress_calls[1][1] == 1
    assert progress_calls[2][0] == "update"
    assert progress_calls[2][1] == 2
    assert progress_calls[-1][0] == "finish"
    assert progress_calls[-1][2]["images"] == 2
    assert (tmp_path / "reports" / "eval" / "end2end_error_analysis.md").exists()


def test_evaluate_end_to_end_manifest_reuses_predictor_sessions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "val.jsonl"
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"fake")
    image_b.write_bytes(b"fake")
    manifest_path.write_text(
        "\n".join(
            [
                '{"image_path": "' + str(image_a) + '", "instances": []}',
                '{"image_path": "' + str(image_b) + '", "instances": []}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    calls = {"detector": 0, "recognizer": 0}

    class _DetectorSession:
        @classmethod
        def from_checkpoint(cls, checkpoint_path):  # noqa: ANN001
            calls["detector"] += 1
            return cls()

    class _RecognizerSession:
        @classmethod
        def from_checkpoint(cls, checkpoint_path):  # noqa: ANN001
            calls["recognizer"] += 1
            return cls()

    def fake_infer_end_to_end_image(**kwargs):  # noqa: ANN001
        return (
            None,
            [
                EndToEndLineResult(
                    line_id="line-0",
                    polygon=[10, 10, 110, 10, 110, 40, 10, 40],
                    text="营业时间",
                    recognition_confidence=0.95,
                    analysis_label="service_info",
                    analysis_confidence=0.9,
                    slots=extract_post_ocr_slots("营业时间"),
                )
            ],
        )

    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline.DetectionPredictorSession",
        _DetectorSession,
    )
    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline.RecognitionPredictorSession",
        _RecognizerSession,
    )
    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline.infer_end_to_end_image",
        fake_infer_end_to_end_image,
    )

    evaluate_end_to_end_manifest(
        manifest_path=manifest_path,
        output_dir=tmp_path / "reports" / "eval",
        detector_checkpoint=tmp_path / "det.pt",
        recognizer_checkpoint=tmp_path / "rec.pt",
    )

    assert calls["detector"] == 1
    assert calls["recognizer"] == 1


def test_sweep_end_to_end_manifest_writes_ranked_summary(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "val.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")

    class _DetectorSession:
        @classmethod
        def from_checkpoint(cls, checkpoint_path):  # noqa: ANN001
            return cls()

    class _RecognizerSession:
        @classmethod
        def from_checkpoint(cls, checkpoint_path):  # noqa: ANN001
            return cls()

    def fake_evaluate_manifest_rows(**kwargs):  # noqa: ANN001
        threshold = kwargs["detector_threshold"]
        min_area = kwargs["min_area"]
        total = 10
        matched = 4 if threshold < 0.5 else 2
        exact = 2 if min_area <= 16 else 1
        image_results = [
            EndToEndBaselineImageResult(
                image_path=Path("sample.png"),
                total_gt=total,
                matched_lines=matched,
                exact_match_lines=exact,
                prediction_texts=["营业时间"] * matched,
                target_texts=["营业时间"] * exact + ["营业"] * max(matched - exact, 0),
            )
        ]
        return aggregate_end_to_end_baseline(image_results), image_results

    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline.DetectionPredictorSession",
        _DetectorSession,
    )
    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline.RecognitionPredictorSession",
        _RecognizerSession,
    )
    monkeypatch.setattr(
        "dltr.pipeline.end_to_end_baseline._evaluate_manifest_rows",
        fake_evaluate_manifest_rows,
    )

    outputs = sweep_end_to_end_manifest(
        manifest_path=manifest_path,
        output_dir=tmp_path / "reports" / "eval",
        detector_checkpoint=tmp_path / "det.pt",
        recognizer_checkpoint=tmp_path / "rec.pt",
        detector_thresholds=[0.4, 0.6],
        min_areas=[16.0, 32.0],
    )

    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
    content = outputs["markdown"].read_text(encoding="utf-8")
    assert "End-to-End Sweep Summary" in content
    assert "0.4000" in content
