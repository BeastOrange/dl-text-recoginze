from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path

from dltr.data.types import DataConfig, DatasetInventory
from dltr.project import ProjectPaths


def collect_inventories(
    project_paths: ProjectPaths,
    config: DataConfig,
) -> dict[str, DatasetInventory]:
    results: dict[str, DatasetInventory] = {}
    for spec in config.datasets:
        dataset_root = project_paths.root / spec.relative_path
        results[spec.name] = scan_dataset_inventory(
            dataset_name=spec.name,
            dataset_root=dataset_root,
            image_extensions=spec.image_extensions,
            label_extensions=spec.label_extensions,
        )
    return results


def scan_dataset_inventory(
    dataset_name: str,
    dataset_root: Path,
    image_extensions: set[str],
    label_extensions: set[str],
) -> DatasetInventory:
    if not dataset_root.exists() or not dataset_root.is_dir():
        return DatasetInventory(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            total_files=0,
            total_images=0,
            total_labels=0,
            image_extension_counts={},
            label_extension_counts={},
            matched_label_images=0,
            missing_label_images=0,
            missing_label_examples=[],
            directory_depth_histogram={},
            image_relpaths=[],
        )

    image_exts = {ext.lower() for ext in image_extensions}
    label_exts = {ext.lower() for ext in label_extensions}
    all_files = _walk_files(dataset_root)

    image_files = [file for file in all_files if file.suffix.lower() in image_exts]
    label_files = [file for file in all_files if file.suffix.lower() in label_exts]

    image_counter = Counter(file.suffix.lower() for file in image_files)
    label_counter = Counter(file.suffix.lower() for file in label_files)

    label_index: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    for label_path in label_files:
        label_index[(label_path.parent, label_path.stem)].append(label_path)

    missing_examples: list[str] = []
    matched = 0
    depth_hist = Counter()
    image_relpaths: list[str] = []

    for image_path in image_files:
        relative = image_path.relative_to(dataset_root)
        image_relpaths.append(relative.as_posix())
        depth_hist[len(relative.parts)] += 1
        if _find_label_matches(
            dataset_root=dataset_root,
            image_path=image_path,
            label_extensions=label_exts,
            label_index=label_index,
        ):
            matched += 1
        elif len(missing_examples) < 20:
            missing_examples.append(relative.as_posix())

    return DatasetInventory(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        total_files=len(all_files),
        total_images=len(image_files),
        total_labels=len(label_files),
        image_extension_counts=dict(image_counter),
        label_extension_counts=dict(label_counter),
        matched_label_images=matched,
        missing_label_images=max(0, len(image_files) - matched),
        missing_label_examples=missing_examples,
        directory_depth_histogram=dict(depth_hist),
        image_relpaths=image_relpaths,
    )


def _walk_files(dataset_root: Path) -> list[Path]:
    files: list[Path] = []
    for root, _, filenames in os.walk(dataset_root, followlinks=True):
        root_path = Path(root)
        for filename in filenames:
            candidate = root_path / filename
            if candidate.is_file():
                files.append(candidate)
    return files


def _find_label_matches(
    dataset_root: Path,
    image_path: Path,
    label_extensions: set[str],
    label_index: dict[tuple[Path, str], list[Path]],
) -> list[Path]:
    direct_matches = label_index.get((image_path.parent, image_path.stem), [])
    if direct_matches:
        return direct_matches

    dataset_specific_matches = _find_dataset_specific_label_matches(
        dataset_root=dataset_root,
        image_path=image_path,
        label_extensions=label_extensions,
    )
    if dataset_specific_matches:
        return dataset_specific_matches

    candidate_dirs = _rects_candidate_label_dirs(dataset_root=dataset_root, image_path=image_path)
    matches: list[Path] = []
    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue
        for extension in label_extensions:
            candidate = candidate_dir / f"{image_path.stem}{extension}"
            if candidate.exists() and candidate.is_file():
                matches.append(candidate)
    return matches


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


def _find_dataset_specific_label_matches(
    dataset_root: Path,
    image_path: Path,
    label_extensions: set[str],
) -> list[Path]:
    annotation_dir = dataset_root / "annotation"
    if not annotation_dir.exists():
        return []

    stem_candidates = [image_path.stem]
    if image_path.stem.startswith("image_"):
        stem_candidates.append(image_path.stem.replace("image_", "gt_img_", 1))

    matches: list[Path] = []
    for stem in stem_candidates:
        for extension in label_extensions:
            candidate = annotation_dir / f"{stem}{extension}"
            if candidate.exists() and candidate.is_file():
                matches.append(candidate)
    return matches
