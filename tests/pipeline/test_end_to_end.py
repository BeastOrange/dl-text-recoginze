import json
from pathlib import Path

import cv2
import numpy as np

from dltr.pipeline.end_to_end import (
    EndToEndLineResult,
    EndToEndPipelineArtifacts,
    run_end_to_end_pipeline,
)
from dltr.post_ocr.slots import extract_post_ocr_slots


def test_run_end_to_end_pipeline_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"fake")

    def fake_pipeline(**_: object) -> EndToEndPipelineArtifacts:
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "result.json"
        markdown_path = output_dir / "result.md"
        preview_path = output_dir / "preview.png"
        payload = {
            "image_path": str(image_path),
            "lines": [
                {
                    "text": "营业时间 09:00-21:00",
                    "analysis_label": "service_info",
                }
            ],
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text("# End-to-End OCR Result\n", encoding="utf-8")
        preview_path.write_bytes(b"png")
        return EndToEndPipelineArtifacts(
            output_dir=output_dir,
            json_path=json_path,
            markdown_path=markdown_path,
            preview_image_path=preview_path,
            line_results=[
                EndToEndLineResult(
                    line_id="line-0",
                    polygon=[10, 10, 100, 10, 100, 40, 10, 40],
                    text="营业时间 09:00-21:00",
                    recognition_confidence=0.75,
                    analysis_label="service_info",
                    analysis_confidence=0.8,
                    slots=extract_post_ocr_slots("营业时间 09:00-21:00"),
                )
            ],
        )

    monkeypatch.setattr(
        "dltr.pipeline.end_to_end._run_pipeline_internal",
        fake_pipeline,
    )

    artifacts = run_end_to_end_pipeline(
        image_path=image_path,
        output_dir=tmp_path / "output",
        detector_checkpoint=tmp_path / "det.pt",
        recognizer_checkpoint=tmp_path / "rec.pt",
    )

    assert artifacts.json_path.exists()
    assert artifacts.markdown_path.exists()
    assert artifacts.preview_image_path.exists()
    assert artifacts.line_results[0].analysis_label == "service_info"


def test_run_end_to_end_pipeline_applies_real_second_pass(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image = np.full((60, 160, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    calls: list[str] = []

    class _DetectorSession:
        @classmethod
        def from_checkpoint(cls, checkpoint_path):  # noqa: ANN001
            return cls()

        def predict_image(self, image, *, threshold, min_area):  # noqa: ANN001
            return [
                type(
                    "Prediction",
                    (),
                    {"polygon": [10, 10, 150, 10, 150, 40, 10, 40], "score": 0.9},
                )()
            ]

    class _RecognizerSession:
        @classmethod
        def from_checkpoint(cls, checkpoint_path):  # noqa: ANN001
            return cls()

        def recognize_images(self, images):  # noqa: ANN001
            calls.append(f"batch-{len(images)}")
            if len(calls) == 1:
                return [type("Prediction", (), {"text": "营", "confidence": 0.42})()]
            return [type("Prediction", (), {"text": "营业时间", "confidence": 0.91})()]

    monkeypatch.setattr(
        "dltr.pipeline.end_to_end.DetectionPredictorSession",
        _DetectorSession,
    )
    monkeypatch.setattr(
        "dltr.pipeline.end_to_end.RecognitionPredictorSession",
        _RecognizerSession,
    )

    artifacts = run_end_to_end_pipeline(
        image_path=image_path,
        output_dir=tmp_path / "output",
        detector_checkpoint=tmp_path / "det.pt",
        recognizer_checkpoint=tmp_path / "rec.pt",
    )

    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    preview = cv2.imread(str(artifacts.preview_image_path))
    assert len(calls) == 2
    assert payload["lines"][0]["text"] == "营业时间"
    assert payload["lines"][0]["second_pass_applied"] is True
    assert payload["runtime_metrics"]["total_latency_ms"] >= 0.0
    assert payload["runtime_metrics"]["fps"] >= 0.0
    assert preview is not None
    assert preview.shape[1] > image.shape[1]


def test_run_end_to_end_pipeline_accepts_reused_sessions(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image = np.full((60, 160, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    class _DetectorSession:
        def predict_image(self, image, *, threshold, min_area):  # noqa: ANN001
            return [
                type(
                    "Prediction",
                    (),
                    {"polygon": [10, 10, 150, 10, 150, 40, 10, 40], "score": 0.9},
                )()
            ]

    class _RecognizerSession:
        def recognize_images(self, images):  # noqa: ANN001
            return [type("Prediction", (), {"text": "营业时间", "confidence": 0.95})()]

    artifacts = run_end_to_end_pipeline(
        image_path=image_path,
        output_dir=tmp_path / "output",
        detector_checkpoint=tmp_path / "det.pt",
        recognizer_checkpoint=tmp_path / "rec.pt",
        detector_session=_DetectorSession(),
        recognizer_session=_RecognizerSession(),
    )

    assert artifacts.line_results[0].text == "营业时间"


def test_run_end_to_end_pipeline_supports_unified_session(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image = np.full((60, 160, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    class _UnifiedSession:
        def infer_image(self, image, *, threshold, min_area):  # noqa: ANN001
            return {
                "preview": image.copy(),
                "line_results": [
                    EndToEndLineResult(
                        line_id="line-0",
                        polygon=[10, 10, 150, 10, 150, 40, 10, 40],
                        text="营业时间",
                        recognition_confidence=0.93,
                        analysis_label="service_info",
                        analysis_confidence=0.88,
                        slots=extract_post_ocr_slots("营业时间"),
                    )
                ],
                "runtime_metrics": {
                    "total_latency_ms": 10.0,
                    "detector_latency_ms": 3.0,
                    "recognizer_latency_ms": 4.0,
                    "second_pass_latency_ms": 1.0,
                    "post_ocr_latency_ms": 2.0,
                    "fps": 100.0,
                },
            }

    artifacts = run_end_to_end_pipeline(
        image_path=image_path,
        output_dir=tmp_path / "output",
        detector_checkpoint=None,
        recognizer_checkpoint=None,
        end2end_session=_UnifiedSession(),
    )

    assert artifacts.line_results[0].text == "营业时间"
    assert artifacts.runtime_metrics["fps"] == 100.0
