import json
from pathlib import Path

from dltr.data.preparation import (
    build_charset_from_manifest,
    combine_recognition_manifests,
    split_manifest,
)


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_combine_recognition_manifests_adds_dataset_source_and_deduplicates(tmp_path: Path) -> None:
    rects_manifest = tmp_path / "rects.jsonl"
    shopsign_manifest = tmp_path / "shopsign.jsonl"
    _write_jsonl(
        rects_manifest,
        [
            {
                "dataset": "rects",
                "image_path": "/a.jpg",
                "label_path": "/a.json",
                "text": "营业时间",
            },
            {
                "dataset": "rects",
                "image_path": "/b.jpg",
                "label_path": "/b.json",
                "text": "电话",
            },
        ],
    )
    _write_jsonl(
        shopsign_manifest,
        [
            {
                "dataset": "shopsign",
                "image_path": "/c.jpg",
                "label_path": "/c.txt",
                "text": "营业时间",
            },
            {
                "dataset": "shopsign",
                "image_path": "/d.jpg",
                "label_path": "/d.txt",
                "text": "折扣",
            },
        ],
    )

    output_path = tmp_path / "combined.jsonl"
    summary = combine_recognition_manifests(
        manifest_paths=[rects_manifest, shopsign_manifest],
        output_path=output_path,
    )

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary.total_rows == 4
    assert summary.dataset_counts["rects"] == 2
    assert summary.dataset_counts["shopsign"] == 2
    assert len(lines) == 4
    assert {line["dataset"] for line in lines} == {"rects", "shopsign"}


def test_build_charset_from_manifest_counts_chinese_and_ascii(tmp_path: Path) -> None:
    manifest = tmp_path / "combined.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "dataset": "rects",
                "image_path": "/a.jpg",
                "label_path": "/a.json",
                "text": "营业时间9:00",
            },
            {
                "dataset": "shopsign",
                "image_path": "/b.jpg",
                "label_path": "/b.txt",
                "text": "MANGO特价",
            },
        ],
    )

    output_path = tmp_path / "charset.txt"
    summary = build_charset_from_manifest(manifest, output_path)

    charset = output_path.read_text(encoding="utf-8").splitlines()
    assert summary.total_characters >= 8
    assert "营" in charset
    assert "M" in charset
    assert ":" in charset


def test_split_manifest_is_deterministic_and_exhaustive(tmp_path: Path) -> None:
    manifest = tmp_path / "combined.jsonl"
    rows = [
        {
            "dataset": "rects",
            "image_path": f"/{index}.jpg",
            "label_path": f"/{index}.json",
            "text": f"文本{index}",
        }
        for index in range(10)
    ]
    _write_jsonl(manifest, rows)

    output_dir = tmp_path / "splits"
    summary = split_manifest(
        manifest_path=manifest,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        seed=7,
    )

    train_lines = (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (output_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()
    test_lines = (output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()

    assert summary.train_rows == 7
    assert summary.val_rows == 2
    assert summary.test_rows == 1
    assert len(train_lines) + len(val_lines) + len(test_lines) == 10
