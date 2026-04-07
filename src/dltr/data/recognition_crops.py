from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dltr.terminal import ProgressBar


@dataclass(frozen=True)
class RecognitionCropSummary:
    split_name: str
    source_rows: int
    emitted_crops: int
    skipped_instances: int
    output_manifest_path: Path


def should_keep_recognition_text(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    if normalized == "###":
        return False
    if "###" in normalized:
        return False
    return True


def extract_recognition_crops_from_detection_manifest(
    *,
    split_name: str,
    detection_manifest_path: Path,
    crop_output_dir: Path,
    output_manifest_path: Path,
    max_samples: int | None = None,
) -> RecognitionCropSummary:
    rows = [
        line
        for line in detection_manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    crop_output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    emitted_crops = 0
    skipped_instances = 0
    written_rows: list[str] = []
    progress = ProgressBar(total=len(rows), description=f"识别裁剪 {split_name}")

    for row_index, raw_line in enumerate(rows):
        if max_samples is not None and emitted_crops >= max_samples:
            break
        payload = json.loads(raw_line)
        image_path = Path(str(payload.get("image_path", "")))
        if not image_path.exists():
            progress.update(
                row_index + 1,
                metrics={"crops": emitted_crops, "skipped": skipped_instances},
            )
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            progress.update(
                row_index + 1,
                metrics={"crops": emitted_crops, "skipped": skipped_instances},
            )
            continue

        dataset = str(payload.get("dataset", "")).strip() or "unknown"
        dataset_crop_dir = crop_output_dir / dataset
        dataset_crop_dir.mkdir(parents=True, exist_ok=True)

        for instance_index, instance in enumerate(payload.get("instances", [])):
            if max_samples is not None and emitted_crops >= max_samples:
                break
            if not isinstance(instance, dict):
                continue
            text = str(instance.get("text", "")).strip()
            if int(instance.get("ignore", 0)) != 0 or not should_keep_recognition_text(text):
                skipped_instances += 1
                continue
            points = [int(value) for value in instance.get("points", [])]
            if not _is_valid_polygon(points):
                skipped_instances += 1
                continue
            crop = _crop_polygon(image, points)
            if crop is None or crop.size == 0:
                skipped_instances += 1
                continue
            crop_path = (
                dataset_crop_dir
                / f"{image_path.stem}_{row_index:05d}_{instance_index:03d}.png"
            )
            cv2.imwrite(str(crop_path), crop)
            written_rows.append(
                json.dumps(
                    {
                        "dataset": dataset,
                        "split": split_name,
                        "source_image_path": str(image_path),
                        "image_path": str(crop_path),
                        "text": text,
                        "instance_index": instance_index,
                    },
                    ensure_ascii=False,
                )
            )
            emitted_crops += 1
        progress.update(
            row_index + 1,
            metrics={"crops": emitted_crops, "skipped": skipped_instances},
        )

    output_manifest_path.write_text(
        "\n".join(written_rows) + ("\n" if written_rows else ""),
        encoding="utf-8",
    )
    progress.finish(metrics={"crops": emitted_crops, "skipped": skipped_instances})
    return RecognitionCropSummary(
        split_name=split_name,
        source_rows=len(rows),
        emitted_crops=emitted_crops,
        skipped_instances=skipped_instances,
        output_manifest_path=output_manifest_path,
    )


def _crop_polygon(image: np.ndarray, points: list[int]) -> np.ndarray | None:
    pts = _polygon_to_quad(points)
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
    cropped = cv2.warpPerspective(image, transform, (target_width, target_height))
    return cropped


def _polygon_to_quad(points: list[int]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 8:
        return pts
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect).astype(np.float32)


def _is_valid_polygon(points: list[int]) -> bool:
    return len(points) >= 8 and len(points) % 2 == 0
