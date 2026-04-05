import json
from pathlib import Path

from dltr.semantic.dataset import load_semantic_samples


def test_load_semantic_samples_filters_missing_text_and_invalid_label(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_id": "a",
                        "text": "营业时间 09:00-21:00",
                        "semantic_class": "service_info",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"source_id": "b", "text": "", "semantic_class": "service_info"},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "source_id": "c",
                        "text": "测试文本",
                        "semantic_class": "unknown",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_semantic_samples(manifest)

    assert len(samples) == 1
    assert samples[0].semantic_class == "service_info"
