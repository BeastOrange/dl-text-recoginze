from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2


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


def load_iiit5k_mat_records(
    *,
    dataset_name: str,
    dataset_root: Path,
    mat_path: Path,
    split: str | None = None,
) -> list[RecognitionSourceRecord]:
    payload = _load_iiit5k_mat_payload(mat_path)
    records: list[RecognitionSourceRecord] = []
    for image_token, text_token in _extract_iiit5k_pairs(payload):
        image_path = _resolve_image_path(
            dataset_root=dataset_root,
            annotation_path=mat_path,
            raw_path=image_token,
        )
        if image_path is None or not text_token.strip():
            continue
        records.append(
            RecognitionSourceRecord(
                dataset=dataset_name,
                image_path=image_path,
                text=text_token.strip(),
                split=split,
            )
        )
    return records


def load_svt_xml_records(
    *,
    dataset_name: str,
    dataset_root: Path,
    xml_path: Path,
    crop_output_dir: Path,
    split: str | None = None,
) -> list[RecognitionSourceRecord]:
    crop_output_dir.mkdir(parents=True, exist_ok=True)
    root = ET.parse(xml_path).getroot()
    records: list[RecognitionSourceRecord] = []
    for image_index, image_node in enumerate(root.findall(".//image")):
        image_token = (image_node.findtext("imageName") or "").strip()
        if not image_token:
            continue
        image_path = _resolve_image_path(
            dataset_root=dataset_root,
            annotation_path=xml_path,
            raw_path=image_token,
        )
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        for word_index, rect_node in enumerate(image_node.findall(".//taggedRectangle")):
            text = (rect_node.findtext("tag") or "").strip()
            if not text:
                continue
            x = int(rect_node.attrib.get("x", "0"))
            y = int(rect_node.attrib.get("y", "0"))
            width = int(rect_node.attrib.get("width", "0"))
            height = int(rect_node.attrib.get("height", "0"))
            crop = image[max(y, 0) : max(y + height, 0), max(x, 0) : max(x + width, 0)]
            if crop.size == 0:
                continue
            crop_path = (
                crop_output_dir / f"{image_path.stem}_{image_index:04d}_{word_index:03d}.png"
            )
            cv2.imwrite(str(crop_path), crop)
            records.append(
                RecognitionSourceRecord(
                    dataset=dataset_name,
                    image_path=crop_path.resolve(),
                    text=text,
                    split=split,
                    label_path=xml_path,
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


def collect_iiit5k_mat_records(
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
        for record in load_iiit5k_mat_records(
            dataset_name=dataset_root.name or "iiit5k",
            dataset_root=dataset_root,
            mat_path=annotation_path,
        )
    ]


def collect_svt_xml_records(
    *,
    dataset_root: Path,
    annotation_path: Path,
    crop_output_dir: Path,
) -> list[RecognitionSourceRecord]:
    return load_svt_xml_records(
        dataset_name=dataset_root.name or "svt",
        dataset_root=dataset_root,
        xml_path=annotation_path,
        crop_output_dir=crop_output_dir,
    )


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
    if not candidate.is_absolute():
        for path in dataset_root.rglob(candidate.name):
            if path.is_file():
                return path.resolve()
    return None


def _load_iiit5k_mat_payload(mat_path: Path) -> dict[str, object]:
    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "IIIT5K `.mat` parsing requires scipy. "
            "Use `uv run --with scipy python scripts/run_dltr.py ...`."
        ) from exc
    return loadmat(mat_path, simplify_cells=True)


def _extract_iiit5k_pairs(payload: object) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    _walk_iiit_payload(payload, pairs)
    return pairs


def _walk_iiit_payload(node: object, pairs: list[tuple[str, str]]) -> None:
    if isinstance(node, dict):
        image_values = _dict_value(node, ("ImgName", "imgName", "ImageName", "imageName"))
        text_values = _dict_value(node, ("GroundTruth", "groundTruth", "Label", "label", "txt"))
        if image_values is not None and text_values is not None:
            image_list = _normalize_to_list(image_values)
            text_list = _normalize_to_list(text_values)
            for image_token, text_token in zip(image_list, text_list, strict=False):
                if image_token and text_token:
                    pairs.append((str(image_token), str(text_token)))
        for value in node.values():
            _walk_iiit_payload(value, pairs)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _walk_iiit_payload(item, pairs)


def _dict_value(payload: dict[str, object], keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _normalize_to_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]
