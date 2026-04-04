import json
from pathlib import Path

from PIL import Image

from dltr.models.detection.dataset import load_detection_samples, rasterize_text_mask


def test_load_detection_samples_filters_missing_images(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color="white").save(image_path)
    manifest_path = tmp_path / "detection.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(image_path),
                        "label_path": "/tmp/a.json",
                        "instances": [
                            {
                                "points": [0, 0, 10, 0, 10, 10, 0, 10],
                                "text": "营业",
                                "ignore": 0,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(tmp_path / "missing.jpg"),
                        "label_path": "/tmp/b.json",
                        "instances": [
                            {
                                "points": [0, 0, 10, 0, 10, 10, 0, 10],
                                "text": "时间",
                                "ignore": 0,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_detection_samples(manifest_path)

    assert len(samples) == 1
    assert samples[0].dataset == "rects"


def test_rasterize_text_mask_marks_polygon_region() -> None:
    mask = rasterize_text_mask(
        image_height=32,
        image_width=32,
        polygons=[[0, 0, 10, 0, 10, 10, 0, 10]],
    )

    assert mask.shape == (32, 32)
    assert float(mask.sum()) > 0.0
