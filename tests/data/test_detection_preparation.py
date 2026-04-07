import json
from pathlib import Path

from dltr.data.detection_preparation import (
    build_detection_manifest,
    split_detection_manifest,
)


def test_build_detection_manifest_supports_rects_json_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "rects"
    image_path = dataset_root / "train" / "img" / "train_ReCTS_000001.jpg"
    label_path = dataset_root / "train" / "gt" / "train_ReCTS_000001.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")
    label_path.write_text(
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

    output_path = tmp_path / "rects_detection.jsonl"
    summary = build_detection_manifest(
        dataset_name="rects",
        dataset_root=dataset_root,
        output_path=output_path,
        image_extensions={".jpg"},
        label_extensions={".json"},
    )

    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert summary.emitted_rows == 1
    assert row["dataset"] == "rects"
    assert len(row["instances"]) == 1
    assert row["instances"][0]["text"] == "营业时间"


def test_build_detection_manifest_supports_shopsign_txt_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "shopsign"
    image_path = dataset_root / "images" / "image_1.jpg"
    label_path = dataset_root / "annotation" / "image_1.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")
    label_path.write_text(
        "0,0,10,0,10,10,0,10,0,MANGO\n"
        "20,20,30,20,30,30,20,30,0,特价",
        encoding="utf-8",
    )

    output_path = tmp_path / "shopsign_detection.jsonl"
    summary = build_detection_manifest(
        dataset_name="shopsign",
        dataset_root=dataset_root,
        output_path=output_path,
        image_extensions={".jpg"},
        label_extensions={".txt"},
    )

    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert summary.emitted_rows == 1
    assert len(row["instances"]) == 2
    assert row["instances"][1]["text"] == "特价"


def test_build_detection_manifest_keeps_curved_polygon_points(tmp_path: Path) -> None:
    dataset_root = tmp_path / "rects"
    image_path = dataset_root / "train" / "img" / "train_curve_000001.jpg"
    label_path = dataset_root / "train" / "gt" / "train_curve_000001.json"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"img")
    label_path.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "points": [0, 0, 8, 0, 12, 4, 12, 10, 6, 12, 0, 10],
                        "transcription": "弯曲文本",
                        "ignore": 0,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "curve_detection.jsonl"
    build_detection_manifest(
        dataset_name="rects",
        dataset_root=dataset_root,
        output_path=output_path,
        image_extensions={".jpg"},
        label_extensions={".json"},
    )

    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["instances"][0]["points"] == [0, 0, 8, 0, 12, 4, 12, 10, 6, 12, 0, 10]


def test_split_detection_manifest_is_exhaustive(tmp_path: Path) -> None:
    manifest_path = tmp_path / "detection.jsonl"
    manifest_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "dataset": "rects",
                    "image_path": f"/{index}.jpg",
                    "label_path": f"/{index}.json",
                    "instances": [
                        {
                            "points": [0, 0, 1, 0, 1, 1, 0, 1],
                            "text": f"文本{index}",
                            "ignore": 0,
                        }
                    ],
                },
                ensure_ascii=False,
            )
            for index in range(10)
        )
        + "\n",
        encoding="utf-8",
    )

    summary = split_detection_manifest(
        manifest_path=manifest_path,
        output_dir=tmp_path / "splits",
        train_ratio=0.6,
        val_ratio=0.2,
        seed=3,
    )

    assert summary.train_rows == 6
    assert summary.val_rows == 2
    assert summary.test_rows == 2
