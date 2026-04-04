from pathlib import Path

from dltr.project import ProjectPaths, discover_project_root, ensure_runtime_dirs


def test_discover_project_root_finds_plan_file() -> None:
    nested = Path("src") / "dltr"
    root = discover_project_root(nested)
    assert root.name == "dl-text-recoginze"
    assert (root / "PLAN.md").exists()


def test_ensure_runtime_dirs_creates_expected_directories(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)

    ensure_runtime_dirs(paths)

    assert paths.change_records.exists()
    assert paths.data_raw.exists()
    assert paths.reports.exists()
