from pathlib import Path

from dltr.data.config import build_default_data_config
from dltr.data.validation import validate_dataset_paths
from dltr.project import ProjectPaths


def test_validate_dataset_paths_reports_missing_required(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)
    paths.data_raw.mkdir(parents=True, exist_ok=True)
    (paths.data_raw / "rects").mkdir(parents=True, exist_ok=True)

    config = build_default_data_config(paths)
    summary = validate_dataset_paths(paths, config)

    missing_required_names = {item.name for item in summary.missing_required}
    assert "rects" not in missing_required_names
    assert "rctw17" not in missing_required_names
    assert "ctr_benchmark_scene_lmdb" in missing_required_names


def test_validate_dataset_paths_accepts_dataset_under_data_root(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)
    dataset_root = tmp_path / "data" / "IIIT5K"
    dataset_root.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        """
datasets:
  - name: iiit5k_test
    relative_path: data/IIIT5K
    required: true
""".strip(),
        encoding="utf-8",
    )
    from dltr.data.config import load_data_config

    config = load_data_config(config_path)
    summary = validate_dataset_paths(paths, config)

    assert summary.ok is True
    assert not summary.invalid_locations
    assert summary.dataset_results[0].within_data_dir is True


def test_validate_dataset_paths_flags_outside_data_root(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)
    outside = tmp_path / "outside_dataset"
    outside.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        """
datasets:
  - name: bad
    relative_path: ../outside_dataset
    required: true
""".strip(),
        encoding="utf-8",
    )
    from dltr.data.config import load_data_config

    config = load_data_config(config_path)
    summary = validate_dataset_paths(paths, config)

    assert len(summary.invalid_locations) == 1
    assert summary.invalid_locations[0].name == "bad"
