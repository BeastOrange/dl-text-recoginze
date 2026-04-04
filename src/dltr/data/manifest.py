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

    image_exts = {ext.lower() for ext in image_extensions}
    label_exts = {ext.lower() for ext in label_extensions}
    images = [
        path
        for path in _walk_files(dataset_root)
        if path.suffix.lower() in image_exts
    ]

    emitted_rows = 0
    skipped_without_label = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in sorted(images):
            label_path = _find_label_path(
                dataset_root=dataset_root,
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
def _find_label_path(
    dataset_root: Path,
    image_path: Path,
    label_extensions: set[str],
) -> Path | None:
    for extension in label_extensions:
        candidate = image_path.with_suffix(extension)
        if candidate.exists() and candidate.is_file():
            return candidate

    if image_path.parent.name == "img":
        for label_dir_name in ("gt", "gt_unicode"):
            label_dir = dataset_root / label_dir_name
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
