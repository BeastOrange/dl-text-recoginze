from __future__ import annotations

import json
import os
from pathlib import Path

from dltr.data.types import ManifestBuildResult


def build_recognition_manifest(
    dataset_name: str,
    dataset_root: Path,
    output_path: Path,
    image_extensions: set[str],
    label_extensions: set[str],
) -> ManifestBuildResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists() or not dataset_root.is_dir():
        output_path.write_text("", encoding="utf-8")
        return ManifestBuildResult(
            dataset_name=dataset_name,
            output_path=output_path,
            scanned_images=0,
            emitted_rows=0,
            skipped_without_label=0,
        )

    source_root = _resolve_manifest_source_root(dataset_root)
    image_exts = {ext.lower() for ext in image_extensions}
    label_exts = {ext.lower() for ext in label_extensions}
    images = [
        path
        for path in _walk_files(source_root)
        if path.suffix.lower() in image_exts
    ]

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
            text = _extract_text(label_path)
            payload = {
                "dataset": dataset_name,
                "image_path": image_path.as_posix(),
                "label_path": label_path.as_posix(),
                "text": text,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            emitted_rows += 1

    return ManifestBuildResult(
        dataset_name=dataset_name,
        output_path=output_path,
        scanned_images=len(images),
        emitted_rows=emitted_rows,
        skipped_without_label=skipped_without_label,
    )


def _resolve_manifest_source_root(dataset_root: Path) -> Path:
    rects_train_root = dataset_root / "train"
    if rects_train_root.exists() and rects_train_root.is_dir():
        return rects_train_root
    return dataset_root
def _find_label_path(
    dataset_root: Path,
    image_path: Path,
    label_extensions: set[str],
) -> Path | None:
    for extension in label_extensions:
        candidate = image_path.with_suffix(extension)
        if candidate.exists() and candidate.is_file():
            return candidate

    dataset_specific = _find_dataset_specific_label_path(
        dataset_root=dataset_root,
        image_path=image_path,
        label_extensions=label_extensions,
    )
    if dataset_specific is not None:
        return dataset_specific

    for label_dir in _rects_candidate_label_dirs(dataset_root=dataset_root, image_path=image_path):
        if not label_dir.exists():
            continue
        for extension in label_extensions:
            candidate = label_dir / f"{image_path.stem}{extension}"
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _walk_files(dataset_root: Path) -> list[Path]:
    files: list[Path] = []
    for root, _, filenames in os.walk(dataset_root, followlinks=True):
        root_path = Path(root)
        for filename in filenames:
            candidate = root_path / filename
            if candidate.is_file():
                files.append(candidate)
    return files


def _extract_text(label_path: Path) -> str:
    suffix = label_path.suffix.lower()
    if suffix == ".txt":
        return _extract_text_from_txt(label_path)
    if suffix == ".json":
        return _extract_text_from_json(label_path)
    return label_path.read_text(encoding="utf-8", errors="ignore").strip()


def _extract_text_from_txt(label_path: Path) -> str:
    content = label_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    tokens: list[str] = []
    for line in content:
        parts = [segment.strip() for segment in line.split(",")]
        if not parts:
            continue
        if len(parts) >= 10 and parts[8] in {"0", "1"}:
            token = ",".join(parts[9:]).strip()
        elif len(parts) >= 9:
            token = ",".join(parts[8:]).strip()
        else:
            token = parts[-1] if len(parts) > 1 else parts[0]
        if token:
            tokens.append(token)
    return " ".join(tokens).strip()


def _extract_text_from_json(label_path: Path) -> str:
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    if isinstance(payload, dict):
        if "lines" in payload and isinstance(payload["lines"], list):
            texts = [
                str(item.get("transcription", "")).strip()
                for item in payload["lines"]
                if isinstance(item, dict)
            ]
            joined = " ".join(token for token in texts if token)
            if joined:
                return joined
        if "chars" in payload and isinstance(payload["chars"], list):
            texts = [
                str(item.get("transcription", "")).strip()
                for item in payload["chars"]
                if isinstance(item, dict)
            ]
            joined = "".join(token for token in texts if token)
            if joined:
                return joined
        value = payload.get("text")
        return str(value).strip() if value is not None else ""
    if isinstance(payload, list):
        texts = [str(item.get("text", "")).strip() for item in payload if isinstance(item, dict)]
        return " ".join([token for token in texts if token]).strip()
    return ""


def _find_dataset_specific_label_path(
    dataset_root: Path,
    image_path: Path,
    label_extensions: set[str],
) -> Path | None:
    annotation_dir = dataset_root / "annotation"
    if not annotation_dir.exists():
        return None

    stem_candidates = [image_path.stem]
    if image_path.stem.startswith("image_"):
        stem_candidates.append(image_path.stem.replace("image_", "gt_img_", 1))

    for stem in stem_candidates:
        for extension in label_extensions:
            candidate = annotation_dir / f"{stem}{extension}"
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _rects_candidate_label_dirs(dataset_root: Path, image_path: Path) -> list[Path]:
    if image_path.parent.name != "img":
        return []

    base_dirs = [
        image_path.parent.parent,
        dataset_root,
    ]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for base_dir in base_dirs:
        for label_dir_name in ("gt", "gt_unicode"):
            candidate = base_dir / label_dir_name
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)
    return candidates
