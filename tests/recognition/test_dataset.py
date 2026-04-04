import json
from pathlib import Path

from dltr.models.recognition.dataset import load_recognition_samples


def test_load_recognition_samples_filters_empty_text_and_missing_images(tmp_path: Path) -> None:
    existing_image = tmp_path / "sample.jpg"
    existing_image.write_bytes(b"img")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(existing_image),
                        "label_path": "/tmp/a.json",
                        "text": "营业时间",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(existing_image),
                        "label_path": "/tmp/b.json",
                        "text": "",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(tmp_path / "missing.jpg"),
                        "label_path": "/tmp/c.json",
                        "text": "电话",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_recognition_samples(manifest_path)

    assert len(samples) == 1
    assert samples[0].text == "营业时间"
