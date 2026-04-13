from __future__ import annotations

from pathlib import Path

from dltr.data.types import DataConfig, DatasetValidationResult, ValidationSummary
from dltr.project import ProjectPaths


def validate_dataset_paths(project_paths: ProjectPaths, config: DataConfig) -> ValidationSummary:
    dataset_results: list[DatasetValidationResult] = []
    data_root = project_paths.data_raw.parent.resolve()
    for spec in config.datasets:
        configured = Path(spec.relative_path)
        resolved = (project_paths.root / configured).resolve()
        within_data_dir = _is_subpath(resolved, data_root)
        exists = resolved.exists()
        issues: list[str] = []
        if not within_data_dir:
            issues.append("dataset path is outside `data/`")
        if spec.required and not exists:
            issues.append("required dataset path does not exist")
        if exists and not resolved.is_dir():
            issues.append("dataset path is not a directory")

        dataset_results.append(
            DatasetValidationResult(
                name=spec.name,
                configured_path=configured,
                resolved_path=resolved,
                required=spec.required,
                within_data_dir=within_data_dir,
                exists=exists and resolved.is_dir(),
                issues=issues,
            )
        )
    return ValidationSummary(dataset_results=dataset_results)


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
