import json
from pathlib import Path

from dltr.cli import main


def test_prepare_detection_command_builds_outputs(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")

    rects_root = tmp_path / "data" / "raw" / "rects" / "train"
    rects_image = rects_root / "img" / "train_ReCTS_000001.jpg"
    rects_label = rects_root / "gt" / "train_ReCTS_000001.json"
    rects_image.parent.mkdir(parents=True, exist_ok=True)
    rects_label.parent.mkdir(parents=True, exist_ok=True)
    rects_image.write_bytes(b"img")
    rects_label.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "points": [0, 0, 10, 0, 10, 10, 0, 10],
                        "transcription": "营业时间",
                        "ignore": 0,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    shopsign_root = tmp_path / "data" / "raw" / "shopsign"
    shopsign_image = shopsign_root / "images" / "image_1.jpg"
    shopsign_label = shopsign_root / "annotation" / "image_1.txt"
    shopsign_image.parent.mkdir(parents=True, exist_ok=True)
    shopsign_label.parent.mkdir(parents=True, exist_ok=True)
    shopsign_image.write_bytes(b"img")
    shopsign_label.write_text("0,0,10,0,10,10,0,10,0,MANGO特价", encoding="utf-8")

    config_dir = tmp_path / "configs" / "data"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "datasets.yaml").write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: rects",
                "    relative_path: data/raw/rects",
                "    required: true",
                "    image_extensions: ['.jpg']",
                "    label_extensions: ['.json']",
                "  - name: shopsign",
                "    relative_path: data/raw/shopsign",
                "    required: true",
                "    image_extensions: ['.jpg']",
                "    label_extensions: ['.txt']",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "data",
            "prepare-detection",
            "--config",
            "configs/data/datasets.yaml",
            "--datasets",
            "rects",
            "shopsign",
        ]
    )

    combined_path = tmp_path / "data" / "processed" / "detection_combined.jsonl"
    train_path = tmp_path / "data" / "processed" / "detection_splits" / "train.jsonl"
    assert exit_code == 0
    assert combined_path.exists()
    assert train_path.exists()
