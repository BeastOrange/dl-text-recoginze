import json
from pathlib import Path

from dltr.data.semantic_preparation import build_semantic_manifests_from_recognition


def test_build_semantic_manifests_from_recognition_writes_splits(tmp_path: Path) -> None:
    recognition_dir = tmp_path / "recognition_splits"
    recognition_dir.mkdir(parents=True, exist_ok=True)
    row = lambda text: json.dumps(  # noqa: E731
        {
            "dataset": "rects",
            "image_path": "/tmp/a.png",
            "text": text,
        },
        ensure_ascii=False,
    )
    (recognition_dir / "train.jsonl").write_text(
        row("营业时间 09:00-21:00 电话13800138000") + "\n" + row("当心高压 危险") + "\n",
        encoding="utf-8",
    )
    (recognition_dir / "val.jsonl").write_text(
        row("开业大促 全场五折 特价") + "\n",
        encoding="utf-8",
    )
    (recognition_dir / "test.jsonl").write_text(row("公告 通知 须知") + "\n", encoding="utf-8")

    output_dir = tmp_path / "semantic"
    outputs = build_semantic_manifests_from_recognition(
        recognition_split_dir=recognition_dir,
        output_dir=output_dir,
    )

    assert outputs["train"].exists()
    assert outputs["val"].exists()
    assert outputs["test"].exists()
    train_row = json.loads(outputs["train"].read_text(encoding="utf-8").splitlines()[0])
    assert train_row["semantic_class"] == "service_info"
