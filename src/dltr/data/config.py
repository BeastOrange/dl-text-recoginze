from __future__ import annotations

from pathlib import Path

import yaml

from dltr.data.types import (
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_LABEL_EXTENSIONS,
    DataConfig,
    DatasetSpec,
)
from dltr.project import ProjectPaths


def build_default_data_config(project_paths: ProjectPaths | None = None) -> DataConfig:
    paths = project_paths or ProjectPaths.from_root()
    base = paths.data_raw
    specs = [
        DatasetSpec(
            name="rctw17",
            relative_path=(base / "rctw17").relative_to(paths.root),
            required=False,
        ),
        DatasetSpec(
            name="rects",
            relative_path=(base / "rects").relative_to(paths.root),
            required=True,
        ),
        DatasetSpec(name="shopsign", relative_path=(base / "shopsign").relative_to(paths.root)),
        DatasetSpec(name="ctw", relative_path=(base / "ctw").relative_to(paths.root)),
        DatasetSpec(name="mtwi", relative_path=(base / "mtwi").relative_to(paths.root)),
        DatasetSpec(
            name="ctr_benchmark_scene_lmdb",
            relative_path=(base / "ctr_benchmark_scene_lmdb").relative_to(paths.root),
            required=True,
        ),
        DatasetSpec(
            name="text_renderer_corpus",
            relative_path=(base / "text_renderer_corpus").relative_to(paths.root),
        ),
        DatasetSpec(
            name="mjsynth",
            relative_path=(base / "mjsynth").relative_to(paths.root),
            manifest_format="mjsynth",
        ),
        DatasetSpec(
            name="iiit5k",
            relative_path=(base / "iiit5k").relative_to(paths.root),
            manifest_format="pairs",
            annotation_path=(base / "iiit5k" / "annotations" / "test.tsv").relative_to(paths.root),
        ),
        DatasetSpec(
            name="svt",
            relative_path=(base / "svt").relative_to(paths.root),
            manifest_format="pairs",
            annotation_path=(base / "svt" / "annotations" / "test.tsv").relative_to(paths.root),
        ),
        DatasetSpec(
            name="icdar13",
            relative_path=(base / "icdar13").relative_to(paths.root),
            manifest_format="icdar_gt",
            annotation_path=(base / "icdar13" / "Challenge2_Test_Task3_GT.txt").relative_to(
                paths.root
            ),
        ),
        DatasetSpec(
            name="icdar15",
            relative_path=(base / "icdar15").relative_to(paths.root),
            manifest_format="icdar_gt",
            annotation_path=(base / "icdar15" / "Challenge4_Test_Task3_GT.txt").relative_to(
                paths.root
            ),
        ),
    ]
    return DataConfig(datasets=specs)


def load_data_config(config_path: Path) -> DataConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    datasets = payload.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("`datasets` must be a list in data config")

    parsed: list[DatasetSpec] = []
    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError("Each dataset config item must be a mapping")
        if "name" not in item or "relative_path" not in item:
            raise ValueError("Each dataset config must include `name` and `relative_path`")
        image_extensions = {str(ext).lower() for ext in item.get("image_extensions", [])}
        label_extensions = {str(ext).lower() for ext in item.get("label_extensions", [])}
        parsed.append(
            DatasetSpec(
                name=str(item["name"]),
                relative_path=Path(str(item["relative_path"])),
                required=bool(item.get("required", False)),
                manifest_format=str(item.get("manifest_format", "sidecar")).strip() or "sidecar",
                annotation_path=(
                    Path(str(item["annotation_path"]))
                    if item.get("annotation_path") is not None
                    else None
                ),
                image_extensions=image_extensions or set(DEFAULT_IMAGE_EXTENSIONS),
                label_extensions=label_extensions or set(DEFAULT_LABEL_EXTENSIONS),
            )
        )
    return DataConfig(datasets=parsed)
