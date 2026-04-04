from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from dltr.models.detection.inference import predict_text_regions
from dltr.models.recognition.inference import recognize_crop
from dltr.semantic.classification import classify_scene_text
from dltr.semantic.slots import SemanticSlots, extract_semantic_slots


@dataclass(frozen=True)
class EndToEndLineResult:
    line_id: str
    polygon: list[int]
    text: str
    recognition_confidence: float
    semantic_class: str
    semantic_confidence: float
    slots: SemanticSlots


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

    for index, detection in enumerate(detections):
        crop = _crop_polygon(original, detection.polygon)
        if crop is None or crop.size == 0:
            continue
        crop_path = crop_dir / f"line_{index:03d}.png"
        cv2.imwrite(str(crop_path), crop)
        recognition = recognize_crop(
            image_path=crop_path,
            checkpoint_path=recognizer_checkpoint,
        )
        semantic = classify_scene_text(recognition.text)
        slots = extract_semantic_slots(recognition.text)
        result = EndToEndLineResult(
            line_id=f"line-{index}",
            polygon=detection.polygon,
            text=recognition.text,
            recognition_confidence=recognition.confidence,
            semantic_class=semantic.semantic_class,
            semantic_confidence=semantic.confidence,
            slots=slots,
        )
        line_results.append(result)
        _draw_polygon(
            preview,
            detection.polygon,
            f"{recognition.text[:12]} | {semantic.semantic_class}",
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
                        "semantic_class": item.semantic_class,
                        "semantic_confidence": item.semantic_confidence,
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
        "| Line | Text | Rec Confidence | Semantic Class | Semantic Confidence |",
        "|---|---|---:|---|---:|",
    ]
    for item in line_results:
        lines.append(
            f"| {item.line_id} | {item.text} | {item.recognition_confidence:.4f} | "
            f"{item.semantic_class} | {item.semantic_confidence:.4f} |"
        )
    return "\n".join(lines) + "\n"
