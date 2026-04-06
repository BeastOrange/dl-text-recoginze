from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from dltr.models.detection.inference import predict_text_regions
from dltr.models.recognition.config import SecondPassConfig
from dltr.models.recognition.inference import recognize_crop
from dltr.models.recognition.refinement import (
    QualitySignals,
    second_pass_reasons,
    should_apply_second_pass,
)
from dltr.models.recognition.trainer import _import_torch
from dltr.post_ocr.classification import analyze_scene_text
from dltr.post_ocr.slots import PostOCRSlots, extract_post_ocr_slots
from dltr.torch_checkpoint import load_torch_checkpoint


@dataclass(frozen=True)
class EndToEndLineResult:
    line_id: str
    polygon: list[int]
    text: str
    recognition_confidence: float
    analysis_label: str
    analysis_confidence: float
    slots: PostOCRSlots
    second_pass_applied: bool = False
    second_pass_reasons: list[str] = field(default_factory=list)
    first_pass_text: str = ""
    first_pass_confidence: float = 0.0
    second_pass_text: str | None = None
    second_pass_confidence: float | None = None


@dataclass(frozen=True)
class EndToEndPipelineArtifacts:
    output_dir: Path
    json_path: Path
    markdown_path: Path
    preview_image_path: Path
    line_results: list[EndToEndLineResult]


def run_end_to_end_pipeline(
    *,
    image_path: Path,
    output_dir: Path,
    detector_checkpoint: Path,
    recognizer_checkpoint: Path,
    detector_threshold: float = 0.5,
    min_area: float = 32.0,
) -> EndToEndPipelineArtifacts:
    return _run_pipeline_internal(
        image_path=image_path,
        output_dir=output_dir,
        detector_checkpoint=detector_checkpoint,
        recognizer_checkpoint=recognizer_checkpoint,
        detector_threshold=detector_threshold,
        min_area=min_area,
    )


def _run_pipeline_internal(
    *,
    image_path: Path,
    output_dir: Path,
    detector_checkpoint: Path,
    recognizer_checkpoint: Path,
    detector_threshold: float = 0.5,
    min_area: float = 32.0,
) -> EndToEndPipelineArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detections = predict_text_regions(
        image_path=image_path,
        checkpoint_path=detector_checkpoint,
        threshold=detector_threshold,
        min_area=min_area,
    )

    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    preview = original.copy()
    line_results: list[EndToEndLineResult] = []
    second_pass_policy = _load_second_pass_policy(recognizer_checkpoint)

    for index, detection in enumerate(detections):
        crop = _crop_polygon(original, detection.polygon)
        if crop is None or crop.size == 0:
            continue
        crop_path = crop_dir / f"line_{index:03d}.png"
        cv2.imwrite(str(crop_path), crop)
        first_pass = recognize_crop(
            image_path=crop_path,
            checkpoint_path=recognizer_checkpoint,
        )
        quality = _estimate_quality_signals(crop)
        reasons = second_pass_reasons(
            first_pass.confidence,
            first_pass.text,
            quality,
            second_pass_policy,
        )
        recognition = first_pass
        second_pass_applied = False
        second_pass_text: str | None = None
        second_pass_confidence: float | None = None
        if should_apply_second_pass(
            first_pass.confidence,
            first_pass.text,
            quality,
            second_pass_policy,
        ):
            second_pass_path = crop_dir / f"line_{index:03d}_second_pass.png"
            enhanced = _apply_second_pass_enhancement(crop)
            cv2.imwrite(str(second_pass_path), enhanced)
            second_pass_prediction = recognize_crop(
                image_path=second_pass_path,
                checkpoint_path=recognizer_checkpoint,
            )
            second_pass_applied = True
            second_pass_text = second_pass_prediction.text
            second_pass_confidence = second_pass_prediction.confidence
            if (
                second_pass_prediction.confidence >= first_pass.confidence
                or not first_pass.text.strip()
            ):
                recognition = second_pass_prediction
        analysis = analyze_scene_text(recognition.text)
        slots = extract_post_ocr_slots(recognition.text)
        result = EndToEndLineResult(
            line_id=f"line-{index}",
            polygon=detection.polygon,
            text=recognition.text,
            recognition_confidence=recognition.confidence,
            analysis_label=analysis.label,
            analysis_confidence=analysis.confidence,
            slots=slots,
            second_pass_applied=second_pass_applied,
            second_pass_reasons=reasons,
            first_pass_text=first_pass.text,
            first_pass_confidence=first_pass.confidence,
            second_pass_text=second_pass_text,
            second_pass_confidence=second_pass_confidence,
        )
        line_results.append(result)
        _draw_polygon(
            preview,
            detection.polygon,
            f"{recognition.text[:12]} | {analysis.label}",
        )

    json_path = output_dir / "end_to_end_result.json"
    markdown_path = output_dir / "end_to_end_result.md"
    preview_path = output_dir / "end_to_end_preview.png"
    json_path.write_text(
        json.dumps(
            {
                "image_path": str(image_path),
                "lines": [
                    {
                        "line_id": item.line_id,
                        "polygon": item.polygon,
                        "text": item.text,
                        "recognition_confidence": item.recognition_confidence,
                        "analysis_label": item.analysis_label,
                        "analysis_confidence": item.analysis_confidence,
                        "second_pass_applied": item.second_pass_applied,
                        "second_pass_reasons": item.second_pass_reasons,
                        "first_pass_text": item.first_pass_text,
                        "first_pass_confidence": item.first_pass_confidence,
                        "second_pass_text": item.second_pass_text,
                        "second_pass_confidence": item.second_pass_confidence,
                        "slots": asdict(item.slots),
                    }
                    for item in line_results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(_build_markdown_report(image_path, line_results), encoding="utf-8")
    cv2.imwrite(str(preview_path), preview)
    return EndToEndPipelineArtifacts(
        output_dir=output_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        preview_image_path=preview_path,
        line_results=line_results,
    )


def _crop_polygon(image: np.ndarray, polygon: list[int]) -> np.ndarray | None:
    pts = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    width_a = np.linalg.norm(pts[2] - pts[3])
    width_b = np.linalg.norm(pts[1] - pts[0])
    height_a = np.linalg.norm(pts[1] - pts[2])
    height_b = np.linalg.norm(pts[0] - pts[3])
    target_width = max(int(round(max(width_a, width_b))), 1)
    target_height = max(int(round(max(height_a, height_b))), 1)
    destination = np.asarray(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(pts, destination)
    return cv2.warpPerspective(image, transform, (target_width, target_height))


def _draw_polygon(image: np.ndarray, polygon: list[int], label: str) -> None:
    pts = np.asarray(polygon, dtype=np.int32).reshape(4, 2)
    cv2.polylines(image, [pts], isClosed=True, color=(0, 180, 0), thickness=2)
    x, y = int(pts[:, 0].min()), int(pts[:, 1].min()) - 6
    cv2.putText(
        image,
        label,
        (max(x, 0), max(y, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 220),
        1,
        cv2.LINE_AA,
    )


def _build_markdown_report(image_path: Path, line_results: list[EndToEndLineResult]) -> str:
    lines = [
        "# End-to-End OCR Result",
        "",
        f"- Image: `{image_path}`",
        f"- Lines: `{len(line_results)}`",
        "",
        "| Line | Text | Rec Confidence | Analysis Label | Analysis Confidence | Second Pass |",
        "|---|---|---:|---|---:|---|",
    ]
    for item in line_results:
        lines.append(
            f"| {item.line_id} | {item.text} | {item.recognition_confidence:.4f} | "
            f"{item.analysis_label} | {item.analysis_confidence:.4f} | "
            f"{'yes' if item.second_pass_applied else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def _estimate_quality_signals(crop: np.ndarray) -> QualitySignals:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur_score = 1.0 / (1.0 + laplacian_var / 100.0)
    contrast_score = min(float(gray.std()) / 64.0, 1.0)
    aspect_ratio = float(gray.shape[1]) / max(float(gray.shape[0]), 1.0)
    return QualitySignals(
        blur_score=blur_score,
        contrast_score=contrast_score,
        aspect_ratio=aspect_ratio,
    )


def _apply_second_pass_enhancement(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(equalized, 1.6, blurred, -0.6, 0.0)


def _load_second_pass_policy(recognizer_checkpoint: Path) -> SecondPassConfig:
    if not recognizer_checkpoint.exists():
        return SecondPassConfig()
    torch = _import_torch()
    checkpoint = load_torch_checkpoint(torch, recognizer_checkpoint, map_location="cpu")
    raw = checkpoint.get("config", {}).get("second_pass", {})
    policy = SecondPassConfig(
        enabled=bool(raw.get("enabled", True)),
        confidence_threshold=float(raw.get("confidence_threshold", 0.78)),
        max_blur_score=float(raw.get("max_blur_score", 0.45)),
        min_contrast_score=float(raw.get("min_contrast_score", 0.35)),
        min_text_length=int(raw.get("min_text_length", 2)),
        min_aspect_ratio=float(raw.get("min_aspect_ratio", 0.2)),
        max_aspect_ratio=float(raw.get("max_aspect_ratio", 20.0)),
    )
    policy.validate()
    return policy
