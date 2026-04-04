from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def discover_project_root(start: Path | None = None) -> Path:
    """Discover the repository root by walking upward until PLAN.md is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "PLAN.md").exists():
            return candidate
    raise FileNotFoundError("Could not discover project root containing PLAN.md")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    src: Path
    configs: Path
    scripts: Path
    tests: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_semantic: Path
    reports: Path
    artifacts: Path
    change_records: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> ProjectPaths:
        repo_root = discover_project_root(root)
        return cls(
            root=repo_root,
            src=repo_root / "src",
            configs=repo_root / "configs",
            scripts=repo_root / "scripts",
            tests=repo_root / "tests",
            data_raw=repo_root / "data" / "raw",
            data_interim=repo_root / "data" / "interim",
            data_processed=repo_root / "data" / "processed",
            data_semantic=repo_root / "data" / "semantic",
            reports=repo_root / "reports",
            artifacts=repo_root / "artifacts",
            change_records=repo_root / "change_records",
        )

    def runtime_dirs(self) -> tuple[Path, ...]:
        return (
            self.configs,
            self.scripts,
            self.tests,
            self.reports,
            self.artifacts,
            self.change_records,
            self.data_raw,
            self.data_interim,
            self.data_processed,
            self.data_semantic,
        )


def ensure_runtime_dirs(paths: ProjectPaths | None = None) -> ProjectPaths:
    project_paths = paths or ProjectPaths.from_root()
    for directory in project_paths.runtime_dirs():
        directory.mkdir(parents=True, exist_ok=True)
    return project_paths
