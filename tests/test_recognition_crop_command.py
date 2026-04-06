from pathlib import Path

from PIL import Image, ImageDraw

from dltr.cli import main


def test_prepare_recognition_crops_command_builds_split_manifests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    detection_dir = tmp_path / "data" / "processed" / "detection_splits"
    detection_dir.mkdir(parents=True, exist_ok=True)

    image_path = tmp_path / "scene.png"
    _write_scene_image(image_path)
    row = (
        f'{{"dataset":"rects","image_path":"{image_path}","label_path":"{image_path.with_suffix(".json")}",'
        '"instances":[{"points":[10,10,110,10,110,50,10,50],"text":"营业时间","ignore":0}]}\n'
    )
    for split in ("train", "val", "test"):
        (detection_dir / f"{split}.jsonl").write_text(row, encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    exit_code = main(["data", "prepare-recognition-crops"])

    assert exit_code == 0
    assert (tmp_path / "data" / "processed" / "recognition_crop_splits" / "train.jsonl").exists()
    assert (tmp_path / "data" / "processed" / "recognition_crop_charset_zh_mixed.txt").exists()


def _write_scene_image(path: Path) -> None:
    image = Image.new("RGB", (128, 128), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 110, 50), outline="black", width=2)
    image.save(path)
