from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_LABEL_EXTENSIONS = {".txt", ".json", ".xml"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    relative_path: Path
    required: bool = False
    image_extensions: set[str] = field(default_factory=lambda: set(DEFAULT_IMAGE_EXTENSIONS))
    label_extensions: set[str] = field(default_factory=lambda: set(DEFAULT_LABEL_EXTENSIONS))


@dataclass(frozen=True)
class DataConfig:
    datasets: list[DatasetSpec]


@dataclass(frozen=True)
class DatasetValidationResult:
    name: str
    configured_path: Path
    resolved_path: Path
    required: bool
    within_data_raw: bool
    exists: bool
    issues: list[str]


@dataclass(frozen=True)
class ValidationSummary:
    dataset_results: list[DatasetValidationResult]

    @property
    def missing_required(self) -> list[DatasetValidationResult]:
        return [item for item in self.dataset_results if item.required and not item.exists]

    @property
    def invalid_locations(self) -> list[DatasetValidationResult]:
        return [item for item in self.dataset_results if not item.within_data_raw]

    @property
    def ok(self) -> bool:
        return not self.missing_required and not self.invalid_locations


@dataclass(frozen=True)
class DatasetInventory:
    dataset_name: str
    dataset_root: Path
    total_files: int
    total_images: int
    total_labels: int
    image_extension_counts: dict[str, int]
    label_extension_counts: dict[str, int]
    matched_label_images: int
    missing_label_images: int
    missing_label_examples: list[str]
    directory_depth_histogram: dict[int, int]
    image_relpaths: list[str]

    @property
    def label_presence_ratio(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.matched_label_images / self.total_images


@dataclass(frozen=True)
class HardCaseMetadata:
    dataset_name: str
    total_images: int
    keyword_hit_counts: dict[str, int]
    keyword_hit_ratio: dict[str, float]
    deep_path_ratio: float
    long_name_ratio: float
    recommendations: list[str]


@dataclass(frozen=True)
class ManifestBuildResult:
    dataset_name: str
    output_path: Path
    scanned_images: int
    emitted_rows: int
    skipped_without_label: int
