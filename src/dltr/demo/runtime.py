from __future__ import annotations

import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from dltr.pipeline.checkpoints import discover_latest_run_dir, resolve_best_checkpoint
from dltr.pipeline.end_to_end import (
    EndToEndLineResult,
    EndToEndPipelineArtifacts,
    run_end_to_end_pipeline,
)
from dltr.post_ocr.classification import analyze_scene_text
from dltr.post_ocr.slots import extract_post_ocr_slots
from dltr.visualization.end_to_end_rendering import render_end_to_end_preview


def resolve_demo_checkpoints(
    *,
    project_root: Path,
    detector_run_dir: Path | None = None,
    recognizer_run_dir: Path | None = None,
) -> dict[str, Path]:
    detection_run = detector_run_dir or discover_latest_run_dir(
        project_root / "artifacts" / "detection"
    )
    recognition_run = recognizer_run_dir or discover_latest_run_dir(
        project_root / "artifacts" / "checkpoints" / "recognition"
    )
    return {
        "detector": resolve_best_checkpoint(detection_run),
        "recognizer": resolve_best_checkpoint(recognition_run),
    }


def run_uploaded_inference(
    *,
    image_bytes: bytes,
    project_root: Path,
    detector_checkpoint: Path,
    recognizer_checkpoint: Path,
) -> EndToEndPipelineArtifacts:
    uploads_dir = project_root / "reports" / "demo_assets" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=uploads_dir, suffix=".png", delete=False) as handle:
        handle.write(image_bytes)
        temp_path = Path(handle.name)

    output_dir = project_root / "reports" / "eval" / "interactive"
    return run_end_to_end_pipeline(
        image_path=temp_path,
        output_dir=output_dir,
        detector_checkpoint=detector_checkpoint,
        recognizer_checkpoint=recognizer_checkpoint,
    )


def _import_paddleocr():
    try:
        import paddleocr  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Run the demo with:\n"
            "uv run --with paddlepaddle --with paddleocr python scripts/run_dltr.py demo --serve"
        ) from exc


def run_paddleocr_e2e_inference(
    *,
    image_bytes: bytes,
    project_root: Path,
) -> EndToEndPipelineArtifacts:
    """
    End-to-end OCR inference using PaddleOCR PP-OCRv5 (English).
    Replaces the self-trained model pipeline which has broken English checkpoints.
    """
    _import_paddleocr()

    import cv2
    import numpy as np
    from paddleocr import PaddleOCR

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")

    uploads_dir = project_root / "reports" / "demo_assets" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=uploads_dir, suffix=".png", delete=False) as handle:
        handle.write(image_bytes)
        temp_path = Path(handle.name)

    output_dir = project_root / "reports" / "eval" / "interactive"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_started_at = time.perf_counter()
    ocr = PaddleOCR(lang="en")
    ocr_started_at = time.perf_counter()
    ocr_result = ocr.ocr(str(temp_path))
    ocr_latency_ms = (time.perf_counter() - ocr_started_at) * 1000.0

    original = cv2.imread(str(temp_path))
    if original is None:
        raise FileNotFoundError(f"Could not read image: {temp_path}")

    line_results: list[EndToEndLineResult] = []
    for idx, page_result in enumerate(ocr_result):
        if page_result is None:
            continue
        # PaddleOCR v3 returns a list of dicts
        dt_polys: list = page_result.get("dt_polys", [])
        rec_texts: list = page_result.get("rec_texts", [])
        rec_scores: list = page_result.get("rec_scores", [])

        for poly, text, score in zip(dt_polys, rec_texts, rec_scores, strict=True):
            text_str = str(text).strip()
            score_float = float(score) if score is not None else 0.0
            # Flatten polygon (N, 2) -> list[int]
            poly_arr: np.ndarray = np.asarray(poly, dtype=np.int32)
            polygon: list[int] = poly_arr.flatten().tolist()

            analysis = analyze_scene_text(text_str)
            slots = extract_post_ocr_slots(text_str)
            line_results.append(
                EndToEndLineResult(
                    line_id=f"line-{idx}",
                    polygon=polygon,
                    text=text_str,
                    recognition_confidence=score_float,
                    analysis_label=analysis.label,
                    analysis_confidence=analysis.confidence,
                    slots=slots,
                )
            )

    preview = render_end_to_end_preview(original, line_results)

    total_latency_ms = (time.perf_counter() - total_started_at) * 1000.0
    fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0
    runtime_metrics = {
        "total_latency_ms": total_latency_ms,
        "ocr_latency_ms": ocr_latency_ms,
        "fps": fps,
    }

    json_path = output_dir / "end_to_end_result.json"
    markdown_path = output_dir / "end_to_end_result.md"
    preview_path = output_dir / "end_to_end_preview.png"

    json_path.write_text(
        __import__("json").dumps(
            {
                "image_path": str(temp_path),
                "engine": "PaddleOCR PP-OCRv5",
                "lines": [
                    {
                        "line_id": item.line_id,
                        "polygon": item.polygon,
                        "text": item.text,
                        "recognition_confidence": item.recognition_confidence,
                        "analysis_label": item.analysis_label,
                        "analysis_confidence": item.analysis_confidence,
                        "slots": asdict(item.slots),
                    }
                    for item in line_results
                ],
                "runtime_metrics": runtime_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        _build_paddleocr_markdown(temp_path, line_results, runtime_metrics),
        encoding="utf-8",
    )
    cv2.imwrite(str(preview_path), preview)
    return EndToEndPipelineArtifacts(
        output_dir=output_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        preview_image_path=preview_path,
        line_results=line_results,
        runtime_metrics=runtime_metrics,
    )


def _build_paddleocr_markdown(
    image_path: Path,
    line_results: list[EndToEndLineResult],
    runtime_metrics: dict[str, float],
) -> str:
    lines = [
        "# End-to-End OCR Result (PaddleOCR PP-OCRv5)",
        "",
        f"- Image: `{image_path}`",
        "- Engine: PaddleOCR PP-OCRv5 (English)",
        f"- Lines: `{len(line_results)}`",
        f"- Total Latency (ms): `{runtime_metrics.get('total_latency_ms', 0.0):.4f}`",
        f"- FPS: `{runtime_metrics.get('fps', 0.0):.4f}`",
        "",
        "| Line | Text | Rec Confidence | Analysis Label | Analysis Confidence |",
        "|---|---|---:|---|---|---:|",
    ]
    for item in line_results:
        lines.append(
            f"| {item.line_id} | {item.text} | {item.recognition_confidence:.4f} | "
            f"{item.analysis_label} | {item.analysis_confidence:.4f} |"
        )
    return "\n".join(lines) + "\n"
