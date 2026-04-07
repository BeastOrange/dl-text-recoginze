from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from dltr.data.manifest import _find_label_path, _walk_files


@dataclass(frozen=True)
class DetectionManifestSummary:
    output_path: Path
    scanned_images: int
    emitted_rows: int
    skipped_without_label: int


@dataclass(frozen=True)
class DetectionSplitSummary:
    output_dir: Path
    train_rows: int
    val_rows: int
    test_rows: int


def build_detection_manifest(
    dataset_name: str,
    dataset_root: Path,
    output_path: Path,
    image_extensions: set[str],
    label_extensions: set[str],
) -> DetectionManifestSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_root.exists() or not dataset_root.is_dir():
        output_path.write_text("", encoding="utf-8")
        return DetectionManifestSummary(
            output_path=output_path,
            scanned_images=0,
            emitted_rows=0,
            skipped_without_label=0,
        )

    source_root = _resolve_detection_source_root(dataset_root)
    image_exts = {ext.lower() for ext in image_extensions}
    label_exts = {ext.lower() for ext in label_extensions}
    images = [path for path in _walk_files(source_root) if path.suffix.lower() in image_exts]

    emitted_rows = 0
    skipped_without_label = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in sorted(images):
            label_path = _find_label_path(
                dataset_root=source_root,
                image_path=image_path,
                label_extensions=label_exts,
            )
            if label_path is None:
                skipped_without_label += 1
                continue
            instances = _extract_instances(label_path)
            if not instances:
                skipped_without_label += 1
                continue
            payload = {
                "dataset": dataset_name,
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "instances": instances,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            emitted_rows += 1

    return DetectionManifestSummary(
        output_path=output_path,
        scanned_images=len(images),
        emitted_rows=emitted_rows,
        skipped_without_label=skipped_without_label,
    )


def combine_detection_manifests(manifest_paths: list[Path], output_path: Path) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_counts: dict[str, int] = {}
    rows: list[str] = []
    seen: set[tuple[str, str]] = set()
    for manifest_path in manifest_paths:
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            dataset = str(payload.get("dataset", "")).strip()
            image_path = str(payload.get("image_path", "")).strip()
            key = (dataset, image_path)
            if key in seen:
                continue
            seen.add(key)
            rows.append(raw_line)
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return dataset_counts


def split_detection_manifest(
    manifest_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DetectionSplitSummary:
    if train_ratio <= 0 or val_ratio < 0:
        raise ValueError("train_ratio must be > 0 and val_ratio must be >= 0")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    rows = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)

    total = len(rows)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    split_rows = {
        "train": rows[:train_end],
        "val": rows[train_end:val_end],
        "test": rows[val_end:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in split_rows.items():
        (output_dir / f"{split_name}.jsonl").write_text(
            "\n".join(split_data) + ("\n" if split_data else ""),
            encoding="utf-8",
        )

    return DetectionSplitSummary(
        output_dir=output_dir,
        train_rows=len(split_rows["train"]),
        val_rows=len(split_rows["val"]),
        test_rows=len(split_rows["test"]),
    )


def _resolve_detection_source_root(dataset_root: Path) -> Path:
    rects_train_root = dataset_root / "train"
    if rects_train_root.exists() and rects_train_root.is_dir():
        return rects_train_root
    return dataset_root


def _extract_instances(label_path: Path) -> list[dict[str, object]]:
    if label_path.suffix.lower() == ".json":
        return _extract_json_instances(label_path)
    if label_path.suffix.lower() == ".txt":
        return _extract_txt_instances(label_path)
    return []


def _extract_json_instances(label_path: Path) -> list[dict[str, object]]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("lines"), list):
        return []

    instances: list[dict[str, object]] = []
    for item in payload["lines"]:
        if not isinstance(item, dict):
            continue
        points = list(item.get("points", []))
        if not _is_valid_polygon(points):
            continue
        instances.append(
            {
                "points": points,
                "text": str(item.get("transcription", "")).strip(),
                "ignore": int(item.get("ignore", 0)),
            }
        )
    return instances


def _extract_txt_instances(label_path: Path) -> list[dict[str, object]]:
    instances: list[dict[str, object]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [segment.strip() for segment in line.split(",")]
        if len(parts) < 9:
            continue
        coord_tokens: list[int] = []
        text_start_index = len(parts)
        ignore = 0
        for index, token in enumerate(parts):
            try:
                coord_tokens.append(int(float(token)))
            except ValueError:
                if index >= 8 and token in {"0", "1"}:
                    ignore = int(token)
                    text_start_index = index + 1
                else:
                    text_start_index = index
                break
        if text_start_index == len(parts):
            text_start_index = len(coord_tokens)
        if len(coord_tokens) >= 9 and len(coord_tokens) % 2 == 1 and coord_tokens[-1] in {0, 1}:
            ignore = coord_tokens[-1]
            coord_tokens = coord_tokens[:-1]
        points = coord_tokens
        if not _is_valid_polygon(points):
            continue
        text = ",".join(parts[text_start_index:]).strip()
        instances.append({"points": points, "text": text, "ignore": ignore})
    return instances


def _is_valid_polygon(points: list[object]) -> bool:
    return len(points) >= 8 and len(points) % 2 == 0
