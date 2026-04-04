"""Data subsystem for Chinese scene text datasets."""

from dltr.data.config import build_default_data_config, load_data_config
from dltr.data.detection_preparation import (
    build_detection_manifest,
    combine_detection_manifests,
    split_detection_manifest,
)
from dltr.data.hardcase import analyze_hardcase_metadata
from dltr.data.inventory import collect_inventories, scan_dataset_inventory
from dltr.data.manifest import build_recognition_manifest
from dltr.data.preparation import (
    build_charset_from_manifest,
    combine_recognition_manifests,
    split_manifest,
)
from dltr.data.reporting import write_eda_markdown_report
from dltr.data.validation import validate_dataset_paths

__all__ = [
    "analyze_hardcase_metadata",
    "build_charset_from_manifest",
    "build_detection_manifest",
    "build_default_data_config",
    "build_recognition_manifest",
    "combine_detection_manifests",
    "combine_recognition_manifests",
    "collect_inventories",
    "load_data_config",
    "scan_dataset_inventory",
    "split_detection_manifest",
    "split_manifest",
    "validate_dataset_paths",
    "write_eda_markdown_report",
]
