from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dltr.models.detection.inference import DetectionPredictorSession
from dltr.models.recognition.inference import RecognitionPredictorSession
from dltr.models.recognition.metrics import compute_recognition_scores
from dltr.pipeline.end_to_end import (
    EndToEndLineResult,
    _load_second_pass_policy,
    infer_end_to_end_image,
)
from dltr.terminal import ProgressBar


@dataclass(frozen=True)
class MatchedLine:
    gt_text: str
    pred_text: str
    iou: float


@dataclass(frozen=True)
class EndToEndMatchResult:
    total_gt: int
    matches: list[MatchedLine]
    missed_gt_texts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EndToEndBaselineImageResult:
    image_path: Path
    total_gt: int
    matched_lines: int
    exact_match_lines: int
    prediction_texts: list[str]
    target_texts: list[str]
    correct_matches: list[MatchedLine] = field(default_factory=list)
    wrong_matches: list[MatchedLine] = field(default_factory=list)
    missed_gt_texts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EndToEndBaselineSummary:
    total_images: int
    total_gt_lines: int
    matched_lines: int
    exact_match_lines: int
    detection_coverage: float
    matched_line_accuracy: float
    system_line_accuracy: float
    matched_cer: float
    matched_ned: float
    matched_mean_edit_distance: float


DEFAULT_SWEEP_THRESHOLDS = (0.3, 0.4, 0.5)
DEFAULT_SWEEP_MIN_AREAS = (16.0, 32.0)


def evaluate_end_to_end_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    detector_checkpoint: Path,
    recognizer_checkpoint: Path,
    max_images: int | None = None,
    detector_threshold: float = 0.5,
    min_area: float = 32.0,
    iou_threshold: float = 0.5,
) -> dict[str, Path]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    detector_session = DetectionPredictorSession.from_checkpoint(detector_checkpoint)
    recognizer_session = RecognitionPredictorSession.from_checkpoint(recognizer_checkpoint)
    second_pass_policy = _load_second_pass_policy(recognizer_checkpoint)
    summary, image_results = _evaluate_manifest_rows(
        rows=rows,
        detector_checkpoint=detector_checkpoint,
        recognizer_checkpoint=recognizer_checkpoint,
        detector_session=detector_session,
        recognizer_session=recognizer_session,
        second_pass_policy=second_pass_policy,
        max_images=max_images,
        detector_threshold=detector_threshold,
        min_area=min_area,
        iou_threshold=iou_threshold,
        description="系统评估",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "end2end_baseline_summary.json"
    markdown_path = output_dir / "end2end_baseline_summary.md"
    analysis_json_path = output_dir / "end2end_error_analysis.json"
    analysis_markdown_path = output_dir / "end2end_error_analysis.md"
    json_path.write_text(
        json.dumps(
            {
                "summary": asdict(summary),
                "images": [asdict(item) for item in image_results],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(_build_markdown_summary(summary), encoding="utf-8")
    analysis_payload = _build_error_analysis_payload(image_results)
    analysis_json_path.write_text(
        json.dumps(analysis_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    analysis_markdown_path.write_text(
        _build_error_analysis_markdown(analysis_payload),
        encoding="utf-8",
    )
    return {
        "json": json_path,
        "markdown": markdown_path,
        "analysis_json": analysis_json_path,
        "analysis_markdown": analysis_markdown_path,
    }


def sweep_end_to_end_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    detector_checkpoint: Path,
    recognizer_checkpoint: Path,
    detector_thresholds: list[float] | None = None,
    min_areas: list[float] | None = None,
    max_images: int | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Path]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved_thresholds = detector_thresholds or list(DEFAULT_SWEEP_THRESHOLDS)
    resolved_min_areas = min_areas or list(DEFAULT_SWEEP_MIN_AREAS)
    detector_session = DetectionPredictorSession.from_checkpoint(detector_checkpoint)
    recognizer_session = RecognitionPredictorSession.from_checkpoint(recognizer_checkpoint)
    second_pass_policy = _load_second_pass_policy(recognizer_checkpoint)
    combos = [
        (threshold, min_area)
        for threshold in resolved_thresholds
        for min_area in resolved_min_areas
    ]
    progress = ProgressBar(total=len(combos), description="参数扫描")
    records: list[dict[str, float | int]] = []
    for index, (threshold, min_area) in enumerate(combos, start=1):
        summary, _ = _evaluate_manifest_rows(
            rows=rows,
            detector_checkpoint=detector_checkpoint,
            recognizer_checkpoint=recognizer_checkpoint,
            detector_session=detector_session,
            recognizer_session=recognizer_session,
            second_pass_policy=second_pass_policy,
            max_images=max_images,
            detector_threshold=threshold,
            min_area=min_area,
            iou_threshold=iou_threshold,
            description=f"扫描 t={threshold:.2f} a={min_area:.0f}",
        )
        records.append(
            {
                "detector_threshold": threshold,
                "min_area": min_area,
                **asdict(summary),
            }
        )
        best = max(
            records,
            key=lambda item: (
                item["system_line_accuracy"],
                item["detection_coverage"],
                -item["matched_cer"],
            ),
        )
        progress.update(
            index,
            metrics={
                "best_thr": best["detector_threshold"],
                "best_area": best["min_area"],
                "best_acc": best["system_line_accuracy"],
            },
        )
    progress.finish(
        metrics={
            "combos": len(records),
            "best_acc": max(item["system_line_accuracy"] for item in records) if records else 0.0,
        }
    )
    records.sort(
        key=lambda item: (
            item["system_line_accuracy"],
            item["detection_coverage"],
            -item["matched_cer"],
        ),
        reverse=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "end2end_sweep_summary.json"
    markdown_path = output_dir / "end2end_sweep_summary.md"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        _build_sweep_markdown(records),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}


def match_predictions_to_ground_truth(
    predictions: list[EndToEndLineResult],
    ground_truth: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> EndToEndMatchResult:
    gt_candidates = [
        item
        for item in ground_truth
        if int(item.get("ignore", 0)) == 0 and len(item.get("points", [])) == 8
    ]
    pairs: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for gt_index, gt_item in enumerate(gt_candidates):
            iou = _polygon_iou(prediction.polygon, list(gt_item["points"]))
            if iou >= iou_threshold:
                pairs.append((iou, pred_index, gt_index))

    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()
    matches: list[MatchedLine] = []
    for iou, pred_index, gt_index in sorted(pairs, reverse=True):
        if pred_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
        matches.append(
            MatchedLine(
                gt_text=str(gt_candidates[gt_index].get("text", "")).strip(),
                pred_text=predictions[pred_index].text,
                iou=iou,
            )
        )

    missed_gt_texts = [
        str(gt_candidates[index].get("text", "")).strip()
        for index in range(len(gt_candidates))
        if index not in matched_ground_truth
    ]
    return EndToEndMatchResult(
        total_gt=len(gt_candidates),
        matches=matches,
        missed_gt_texts=missed_gt_texts,
    )


def aggregate_end_to_end_baseline(
    image_results: list[EndToEndBaselineImageResult],
) -> EndToEndBaselineSummary:
    total_gt_lines = sum(item.total_gt for item in image_results)
    matched_lines = sum(item.matched_lines for item in image_results)
    exact_match_lines = sum(item.exact_match_lines for item in image_results)
    prediction_texts = [text for item in image_results for text in item.prediction_texts]
    target_texts = [text for item in image_results for text in item.target_texts]
    if prediction_texts and target_texts:
        recognition = compute_recognition_scores(prediction_texts, target_texts)
        matched_cer = recognition.cer
        matched_ned = recognition.ned
        matched_mean_edit_distance = recognition.mean_edit_distance
    else:
        matched_cer = 1.0
        matched_ned = 1.0
        matched_mean_edit_distance = 0.0

    return EndToEndBaselineSummary(
        total_images=len(image_results),
        total_gt_lines=total_gt_lines,
        matched_lines=matched_lines,
        exact_match_lines=exact_match_lines,
        detection_coverage=_safe_div(matched_lines, total_gt_lines),
        matched_line_accuracy=_safe_div(exact_match_lines, matched_lines),
        system_line_accuracy=_safe_div(exact_match_lines, total_gt_lines),
        matched_cer=matched_cer,
        matched_ned=matched_ned,
        matched_mean_edit_distance=matched_mean_edit_distance,
    )


def _polygon_iou(left: list[int], right: list[int]) -> float:
    left_pts = np.asarray(left, dtype=np.float32).reshape(4, 2)
    right_pts = np.asarray(right, dtype=np.float32).reshape(4, 2)
    min_x = int(np.floor(min(left_pts[:, 0].min(), right_pts[:, 0].min())))
    min_y = int(np.floor(min(left_pts[:, 1].min(), right_pts[:, 1].min())))
    max_x = int(np.ceil(max(left_pts[:, 0].max(), right_pts[:, 0].max())))
    max_y = int(np.ceil(max(left_pts[:, 1].max(), right_pts[:, 1].max())))
    width = max(max_x - min_x + 3, 1)
    height = max(max_y - min_y + 3, 1)

    left_mask = np.zeros((height, width), dtype=np.uint8)
    right_mask = np.zeros((height, width), dtype=np.uint8)
    origin = np.asarray([[min_x - 1, min_y - 1]], dtype=np.float32)
    left_shifted = (left_pts - origin).astype(np.int32)
    right_shifted = (right_pts - origin).astype(np.int32)
    cv2.fillPoly(left_mask, [left_shifted], 1)
    cv2.fillPoly(right_mask, [right_shifted], 1)
    intersection = float(np.logical_and(left_mask, right_mask).sum())
    union = float(np.logical_or(left_mask, right_mask).sum())
    return _safe_div(intersection, union)


def _build_markdown_summary(summary: EndToEndBaselineSummary) -> str:
    lines = [
        "# End-to-End Baseline Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total_images | {summary.total_images} |",
        f"| total_gt_lines | {summary.total_gt_lines} |",
        f"| matched_lines | {summary.matched_lines} |",
        f"| exact_match_lines | {summary.exact_match_lines} |",
        f"| detection_coverage | {summary.detection_coverage:.6f} |",
        f"| matched_line_accuracy | {summary.matched_line_accuracy:.6f} |",
        f"| system_line_accuracy | {summary.system_line_accuracy:.6f} |",
        f"| matched_cer | {summary.matched_cer:.6f} |",
        f"| matched_ned | {summary.matched_ned:.6f} |",
        f"| matched_mean_edit_distance | {summary.matched_mean_edit_distance:.6f} |",
    ]
    return "\n".join(lines) + "\n"


def _evaluate_manifest_rows(
    *,
    rows: list[dict[str, Any]],
    detector_checkpoint: Path,
    recognizer_checkpoint: Path,
    detector_session: DetectionPredictorSession,
    recognizer_session: RecognitionPredictorSession,
    second_pass_policy: Any,
    max_images: int | None,
    detector_threshold: float,
    min_area: float,
    iou_threshold: float,
    description: str,
) -> tuple[EndToEndBaselineSummary, list[EndToEndBaselineImageResult]]:
    total_rows = min(len(rows), max_images) if max_images is not None else len(rows)
    progress = ProgressBar(total=total_rows, description=description)
    image_results: list[EndToEndBaselineImageResult] = []
    for index, payload in enumerate(rows):
        if max_images is not None and index >= max_images:
            break
        image_path = Path(str(payload.get("image_path", "")))
        instances = list(payload.get("instances", []))
        _, line_results = infer_end_to_end_image(
            image_path=image_path,
            detector_checkpoint=detector_checkpoint,
            recognizer_checkpoint=recognizer_checkpoint,
            detector_threshold=detector_threshold,
            min_area=min_area,
            detector_session=detector_session,
            recognizer_session=recognizer_session,
            second_pass_policy=second_pass_policy,
        )
        match_result = match_predictions_to_ground_truth(
            line_results,
            instances,
            iou_threshold=iou_threshold,
        )
        correct_matches = [item for item in match_result.matches if item.pred_text == item.gt_text]
        wrong_matches = [item for item in match_result.matches if item.pred_text != item.gt_text]
        prediction_texts = [item.pred_text for item in match_result.matches]
        target_texts = [item.gt_text for item in match_result.matches]
        image_results.append(
            EndToEndBaselineImageResult(
                image_path=image_path,
                total_gt=match_result.total_gt,
                matched_lines=len(match_result.matches),
                exact_match_lines=len(correct_matches),
                prediction_texts=prediction_texts,
                target_texts=target_texts,
                correct_matches=correct_matches,
                wrong_matches=wrong_matches,
                missed_gt_texts=match_result.missed_gt_texts,
            )
        )
        partial = aggregate_end_to_end_baseline(image_results)
        progress.update(
            len(image_results),
            metrics={
                "images": len(image_results),
                "matched": partial.matched_lines,
                "coverage": partial.detection_coverage,
                "exact": partial.exact_match_lines,
            },
        )
    summary = aggregate_end_to_end_baseline(image_results)
    progress.finish(
        metrics={
            "images": summary.total_images,
            "matched": summary.matched_lines,
            "coverage": summary.detection_coverage,
            "exact": summary.exact_match_lines,
        }
    )
    return summary, image_results


def _build_error_analysis_payload(
    image_results: list[EndToEndBaselineImageResult],
) -> dict[str, Any]:
    missed = [
        {
            "image_path": str(item.image_path),
            "missed_count": len(item.missed_gt_texts),
            "missed_gt_texts": item.missed_gt_texts[:5],
        }
        for item in image_results
        if item.missed_gt_texts
    ]
    wrong = [
        {
            "image_path": str(item.image_path),
            "gt_text": match.gt_text,
            "pred_text": match.pred_text,
            "iou": match.iou,
        }
        for item in image_results
        for match in item.wrong_matches
    ]
    correct = [
        {
            "image_path": str(item.image_path),
            "gt_text": match.gt_text,
            "pred_text": match.pred_text,
            "iou": match.iou,
        }
        for item in image_results
        for match in item.correct_matches
    ]
    return {
        "missed_detection": missed[:50],
        "wrong_recognition": wrong[:50],
        "correct_samples": correct[:30],
        "counts": {
            "missed_detection_images": len(missed),
            "wrong_recognition_matches": len(wrong),
            "correct_matches": len(correct),
        },
    }


def _build_error_analysis_markdown(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    lines = [
        "# End-to-End Error Analysis",
        "",
        f"- Missed detection images: `{counts['missed_detection_images']}`",
        f"- Wrong recognition matches: `{counts['wrong_recognition_matches']}`",
        f"- Correct matches: `{counts['correct_matches']}`",
        "",
        "## Missed Detections",
        "",
        "| Image | Missed Count | GT Examples |",
        "|---|---:|---|",
    ]
    for item in payload["missed_detection"]:
        lines.append(
            f"| {item['image_path']} | {item['missed_count']} | "
            f"{', '.join(item['missed_gt_texts']) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Wrong Recognitions",
            "",
            "| Image | GT | Pred | IoU |",
            "|---|---|---|---:|",
        ]
    )
    for item in payload["wrong_recognition"]:
        lines.append(
            f"| {item['image_path']} | {item['gt_text']} | "
            f"{item['pred_text']} | {item['iou']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Correct Samples",
            "",
            "| Image | Text | IoU |",
            "|---|---|---:|",
        ]
    )
    for item in payload["correct_samples"]:
        lines.append(
            f"| {item['image_path']} | {item['gt_text']} | {item['iou']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _build_sweep_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# End-to-End Sweep Summary",
        "",
        "| Threshold | Min Area | Coverage | System Acc | Matched Acc | CER |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            f"| {item['detector_threshold']:.4f} | {item['min_area']:.1f} | "
            f"{item['detection_coverage']:.6f} | {item['system_line_accuracy']:.6f} | "
            f"{item['matched_line_accuracy']:.6f} | {item['matched_cer']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
