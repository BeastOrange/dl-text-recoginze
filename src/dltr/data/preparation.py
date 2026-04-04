from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CombinedManifestSummary:
    output_path: Path
    total_rows: int
    dataset_counts: dict[str, int]


@dataclass(frozen=True)
class CharsetBuildSummary:
    output_path: Path
    total_characters: int
    unique_characters: int


@dataclass(frozen=True)
class ManifestSplitSummary:
    output_dir: Path
    train_rows: int
    val_rows: int
    test_rows: int


def combine_recognition_manifests(
    manifest_paths: list[Path],
    output_path: Path,
) -> CombinedManifestSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    dataset_counts: Counter[str] = Counter()

    for manifest_path in manifest_paths:
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            dataset = str(payload.get("dataset", "")).strip()
            image_path = str(payload.get("image_path", "")).strip()
            text = str(payload.get("text", "")).strip()
            key = (dataset, image_path, text)
            if key in seen:
                continue
            seen.add(key)
            rows.append(payload)
            dataset_counts[dataset] += 1

    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return CombinedManifestSummary(
        output_path=output_path,
        total_rows=len(rows),
        dataset_counts=dict(dataset_counts),
    )


def build_charset_from_manifest(
    manifest_path: Path,
    output_path: Path,
    min_frequency: int = 1,
) -> CharsetBuildSummary:
    if min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")

    counter: Counter[str] = Counter()
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        text = str(payload.get("text", ""))
        for character in text:
            if character.strip():
                counter[character] += 1

    charset = sorted(character for character, count in counter.items() if count >= min_frequency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(charset) + ("\n" if charset else ""), encoding="utf-8")
    return CharsetBuildSummary(
        output_path=output_path,
        total_characters=sum(counter.values()),
        unique_characters=len(charset),
    )


def split_manifest(
    manifest_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> ManifestSplitSummary:
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

    return ManifestSplitSummary(
        output_dir=output_dir,
        train_rows=len(split_rows["train"]),
        val_rows=len(split_rows["val"]),
        test_rows=len(split_rows["test"]),
    )
