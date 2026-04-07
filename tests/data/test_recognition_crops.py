import json
from pathlib import Path

from PIL import Image, ImageDraw

from dltr.data.recognition_crops import (
    extract_recognition_crops_from_detection_manifest,
    should_keep_recognition_text,
)


def test_should_keep_recognition_text_filters_invalid_targets() -> None:
    assert should_keep_recognition_text("营业时间") is True
    assert should_keep_recognition_text("") is False
    assert should_keep_recognition_text("###") is False
    assert should_keep_recognition_text("MANGO ###") is False


def test_extract_recognition_crops_from_detection_manifest_writes_crop_manifest(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    _write_scene_image(image_path)
    manifest_path = tmp_path / "train.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "rects",
                "image_path": str(image_path),
                "label_path": str(image_path.with_suffix(".json")),
                "instances": [
                    {
                        "points": [10, 10, 110, 10, 110, 50, 10, 50],
                        "text": "营业时间",
                        "ignore": 0,
                    },
                    {
                        "points": [10, 60, 110, 60, 110, 100, 10, 100],
                        "text": "###",
                        "ignore": 0,
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "crops" / "train"
    output_manifest = tmp_path / "recognition_train.jsonl"
    summary = extract_recognition_crops_from_detection_manifest(
        split_name="train",
        detection_manifest_path=manifest_path,
        crop_output_dir=output_dir,
        output_manifest_path=output_manifest,
    )

    rows = [json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines()]
    assert summary.source_rows == 1
    assert summary.emitted_crops == 1
    assert rows[0]["text"] == "营业时间"
    assert Path(rows[0]["image_path"]).exists()


def test_extract_recognition_crops_supports_polygon_with_more_than_four_points(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    _write_scene_image(image_path)
    manifest_path = tmp_path / "curve_train.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "rects",
                "image_path": str(image_path),
                "label_path": str(image_path.with_suffix(".json")),
                "instances": [
                    {
                        "points": [10, 10, 60, 10, 110, 20, 110, 50, 60, 50, 10, 40],
                        "text": "弯曲文本",
                        "ignore": 0,
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    output_manifest = tmp_path / "curve_recognition_train.jsonl"
    summary = extract_recognition_crops_from_detection_manifest(
        split_name="train",
        detection_manifest_path=manifest_path,
        crop_output_dir=tmp_path / "crops",
        output_manifest_path=output_manifest,
    )

    rows = [json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines()]
    assert summary.emitted_crops == 1
    assert rows[0]["text"] == "弯曲文本"
    assert Path(rows[0]["image_path"]).exists()


def _write_scene_image(path: Path) -> None:
    image = Image.new("RGB", (128, 128), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 110, 50), outline="black", width=2)
    draw.rectangle((10, 60, 110, 100), outline="black", width=2)
    image.save(path)
