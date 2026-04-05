from pathlib import Path

from dltr.visualization.hardcase_reports import build_hardcase_overview


def test_build_hardcase_overview_writes_outputs(tmp_path: Path) -> None:
    datasets_yaml = tmp_path / "datasets.yaml"
    datasets_yaml.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: rects",
                "    relative_path: data/raw/rects",
                "    required: true",
                "    image_extensions: ['.jpg']",
                "    label_extensions: ['.json']",
            ]
        ),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "data" / "raw" / "rects" / "train" / "img"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "small_blur.jpg").write_bytes(b"img")

    outputs = build_hardcase_overview(
        config_path=datasets_yaml,
        output_dir=tmp_path / "reports" / "hardcase",
        project_root=tmp_path,
    )

    assert outputs["markdown"].exists()
    assert outputs["png"].exists()
