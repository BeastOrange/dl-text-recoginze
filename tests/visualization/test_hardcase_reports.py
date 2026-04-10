from pathlib import Path

from dltr.visualization.hardcase_reports import (
    _resolve_hardcase_upper_bound,
    build_hardcase_overview,
)


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
                "  - name: transformer_detector_crop_cn_scene_4090_multitask",
                "    relative_path: data/raw/transformer_detector_crop_cn_scene_4090_multitask",
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
    second_root = (
        tmp_path
        / "data"
        / "raw"
        / "transformer_detector_crop_cn_scene_4090_multitask"
        / "train"
        / "img"
    )
    second_root.mkdir(parents=True, exist_ok=True)
    (second_root / "dense_rotate.jpg").write_bytes(b"img")

    outputs = build_hardcase_overview(
        config_path=datasets_yaml,
        output_dir=tmp_path / "reports" / "hardcase",
        project_root=tmp_path,
    )

    assert outputs["markdown"].exists()
    assert outputs["png"].exists()
    assert outputs["png"].stat().st_size > 0


def test_resolve_hardcase_upper_bound_handles_all_zero_case() -> None:
    assert _resolve_hardcase_upper_bound(0.0) == 0.05
    assert _resolve_hardcase_upper_bound(0.02) >= 0.05
    assert _resolve_hardcase_upper_bound(0.5) > 0.05
