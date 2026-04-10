from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecognitionSourceRecord:
    dataset: str
    image_path: Path
    text: str
    split: str | None = None
    label_path: Path | None = None


def parse_mjsynth_label_from_stem(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 3:
        return ""

    if parts[0].isdigit() and parts[-1].isdigit():
        return "_".join(parts[1:-1]).strip()
    if parts[-2].isdigit() and parts[-1].isdigit():
        return "_".join(parts[:-2]).strip()
    return "_".join(parts[1:-1]).strip()


def load_mjsynth_records(
    *,
    dataset_name: str,
    dataset_root: Path,
    image_extensions: set[str],
    split: str | None = None,
) -> list[RecognitionSourceRecord]:
    image_exts = {ext.lower() for ext in image_extensions}
    records: list[RecognitionSourceRecord] = []
    for image_path in sorted(dataset_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
            continue
        label = parse_mjsynth_label_from_stem(image_path.stem)
        if not label:
            continue
        records.append(
            RecognitionSourceRecord(
                dataset=dataset_name,
                image_path=image_path.resolve(),
                text=label,
                split=split,
            )
        )
    return records


def load_pairs_records(
    *,
    dataset_name: str,
    dataset_root: Path,
    pairs_path: Path,
    split: str | None = None,
) -> list[RecognitionSourceRecord]:
    records: list[RecognitionSourceRecord] = []
    for raw_line in pairs_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "\t" in line:
            image_token, text_token = line.split("\t", maxsplit=1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            image_token, text_token = parts
        image_path = _resolve_image_path(
            dataset_root=dataset_root,
            annotation_path=pairs_path,
            raw_path=image_token,
        )
        text = text_token.strip()
        if image_path is None or not text:
            continue
        records.append(
            RecognitionSourceRecord(
                dataset=dataset_name,
                image_path=image_path,
                text=text,
                split=split,
            )
        )
    return records


def load_icdar_gt_records(
    *,
    dataset_name: str,
    dataset_root: Path,
    gt_path: Path,
    split: str | None = None,
) -> list[RecognitionSourceRecord]:
    records: list[RecognitionSourceRecord] = []
    for raw_line in gt_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        image_token, text_token = _split_icdar_gt_line(line)
        if image_token is None or text_token is None:
            continue
        image_path = _resolve_image_path(
            dataset_root=dataset_root,
            annotation_path=gt_path,
            raw_path=image_token,
        )
        text = _strip_wrapping_quotes(text_token.strip())
        if image_path is None or not text:
            continue
        records.append(
            RecognitionSourceRecord(
                dataset=dataset_name,
                image_path=image_path,
                text=text,
                split=split,
            )
        )
    return records


def collect_mjsynth_records(
    *,
    dataset_root: Path,
    image_extensions: set[str],
) -> list[RecognitionSourceRecord]:
    return load_mjsynth_records(
        dataset_name="mjsynth",
        dataset_root=dataset_root,
        image_extensions=image_extensions,
    )


def collect_pairs_records(
    *,
    dataset_root: Path,
    annotation_path: Path,
) -> list[RecognitionSourceRecord]:
    return [
        RecognitionSourceRecord(
            dataset=record.dataset,
            image_path=record.image_path,
            text=record.text,
            split=record.split,
            label_path=annotation_path,
        )
        for record in load_pairs_records(
            dataset_name=dataset_root.name or "pairs",
            dataset_root=dataset_root,
            pairs_path=annotation_path,
        )
    ]


def collect_icdar_gt_records(
    *,
    dataset_root: Path,
    annotation_path: Path,
) -> list[RecognitionSourceRecord]:
    return [
        RecognitionSourceRecord(
            dataset=record.dataset,
            image_path=record.image_path,
            text=record.text,
            split=record.split,
            label_path=annotation_path,
        )
        for record in load_icdar_gt_records(
            dataset_name=dataset_root.name or "icdar_gt",
            dataset_root=dataset_root,
            gt_path=annotation_path,
        )
    ]


def _split_icdar_gt_line(line: str) -> tuple[str | None, str | None]:
    if "," not in line:
        return None, None
    image_token, text_token = line.split(",", maxsplit=1)
    return image_token.strip(), text_token.strip()


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def _resolve_image_path(
    *,
    dataset_root: Path,
    annotation_path: Path,
    raw_path: str,
) -> Path | None:
    candidate = Path(raw_path)
    probe_paths: list[Path] = []
    if candidate.is_absolute():
        probe_paths.append(candidate)
    else:
        probe_paths.append(dataset_root / candidate)
        probe_paths.append(annotation_path.parent / candidate)
    for path in probe_paths:
        if path.exists() and path.is_file():
            return path.resolve()
    return None
