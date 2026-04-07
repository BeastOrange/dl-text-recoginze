from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionInstance:
    points: list[int]
    text: str
    ignore: int


@dataclass(frozen=True)
class DetectionSample:
    dataset: str
    image_path: Path
    label_path: Path
    instances: list[DetectionInstance]


def load_detection_samples(manifest_path: str | Path) -> list[DetectionSample]:
    path = Path(manifest_path)
    samples: list[DetectionSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        image_path = Path(str(payload.get("image_path", "")))
        if not image_path.exists():
            continue
        raw_instances = payload.get("instances", [])
        instances = [
            DetectionInstance(
                points=[int(value) for value in item.get("points", [])],
                text=str(item.get("text", "")).strip(),
                ignore=int(item.get("ignore", 0)),
            )
            for item in raw_instances
            if isinstance(item, dict) and _is_valid_polygon(item.get("points", []))
        ]
        if not instances:
            continue
        samples.append(
            DetectionSample(
                dataset=str(payload.get("dataset", "")).strip(),
                image_path=image_path,
                label_path=Path(str(payload.get("label_path", ""))),
                instances=instances,
            )
        )
    return samples


def rasterize_text_mask(
    *,
    image_height: int,
    image_width: int,
    polygons: list[list[int]],
) -> np.ndarray:
    mask = np.zeros((image_height, image_width), dtype=np.float32)
    for polygon in polygons:
        if not _is_valid_polygon(polygon):
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], 0, image_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, image_height - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], color=1.0)
    return mask


def _is_valid_polygon(points: list[object]) -> bool:
    return len(points) >= 8 and len(points) % 2 == 0
