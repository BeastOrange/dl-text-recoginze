"""Data subsystem for Chinese scene text datasets."""

from dltr.data.config import build_default_data_config, load_data_config
from dltr.data.hardcase import analyze_hardcase_metadata
from dltr.data.inventory import collect_inventories, scan_dataset_inventory
from dltr.data.manifest import build_recognition_manifest
from dltr.data.reporting import write_eda_markdown_report
from dltr.data.validation import validate_dataset_paths

__all__ = [
    "analyze_hardcase_metadata",
    "build_default_data_config",
    "build_recognition_manifest",
    "collect_inventories",
    "load_data_config",
    "scan_dataset_inventory",
    "validate_dataset_paths",
    "write_eda_markdown_report",
]
