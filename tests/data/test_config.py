from pathlib import Path

from dltr.data.config import build_default_data_config, load_data_config
from dltr.project import ProjectPaths


def test_load_data_config_reads_dataset_items(tmp_path: Path) -> None:
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        """
datasets:
  - name: rctw17
    relative_path: data/raw/rctw17
    required: true
    image_extensions: [".jpg"]
    label_extensions: [".txt"]
""".strip(),
        encoding="utf-8",
    )

    config = load_data_config(config_path)

    assert len(config.datasets) == 1
    assert config.datasets[0].name == "rctw17"
    assert config.datasets[0].required is True
    assert ".jpg" in config.datasets[0].image_extensions


def test_build_default_data_config_contains_required_datasets(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)

    config = build_default_data_config(paths)

    required_names = {item.name for item in config.datasets if item.required}
    assert "rects" in required_names
    assert "ctr_benchmark_scene_lmdb" in required_names
    assert "rctw17" not in required_names


def test_build_default_data_config_contains_english_benchmark_datasets(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)

    config = build_default_data_config(paths)

    dataset_names = {item.name for item in config.datasets}
    assert {"mjsynth", "iiit5k", "svt", "icdar13", "icdar15"} <= dataset_names


def test_load_data_config_reads_manifest_format_and_annotation_path(tmp_path: Path) -> None:
    config_path = tmp_path / "datasets.yaml"
    config_path.write_text(
        """
datasets:
  - name: mjsynth
    relative_path: data/raw/mjsynth
    manifest_format: mjsynth
  - name: icdar13
    relative_path: data/raw/icdar13
    manifest_format: icdar_gt
    annotation_path: data/raw/icdar13/annotations/test.txt
""".strip(),
        encoding="utf-8",
    )

    config = load_data_config(config_path)

    assert config.datasets[0].manifest_format == "mjsynth"
    assert config.datasets[0].annotation_path is None
    assert config.datasets[1].manifest_format == "icdar_gt"
    assert config.datasets[1].annotation_path == Path("data/raw/icdar13/annotations/test.txt")
