import json

import pytest

from dltr.cli import main

torch = pytest.importorskip("torch")


def test_train_semantic_runs_smoke(tmp_path, monkeypatch) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    train_manifest = tmp_path / "data" / "semantic" / "cn_scenetext_sem" / "train.jsonl"
    val_manifest = tmp_path / "data" / "semantic" / "cn_scenetext_sem" / "val.jsonl"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source_id": "a",
            "text": "营业时间 09:00-21:00 电话13800138000",
            "semantic_class": "service_info",
        },
        {"source_id": "b", "text": "开业大促 全场五折 特价", "semantic_class": "advertisement"},
        {"source_id": "c", "text": "当心高压 危险", "semantic_class": "traffic_or_warning"},
    ]
    train_manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    val_manifest.write_text(train_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    config = tmp_path / "configs" / "semantic" / "baseline.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "\n".join(
            [
                "experiment_name: semantic_cmd_smoke",
                "model_name: char_linear",
                "label_set:",
                "  - shop_sign",
                "  - advertisement",
                "  - public_notice",
                "  - traffic_or_warning",
                "  - service_info",
                "  - other",
                "dataset_manifest: data/semantic/cn_scenetext_sem/train.jsonl",
                "validation_manifest: data/semantic/cn_scenetext_sem/val.jsonl",
                "output_dir: artifacts/checkpoints/semantic/semantic_cmd_smoke",
                "epochs: 1",
                "batch_size: 2",
                "learning_rate: 0.001",
                "max_length: 256",
                "device: cpu",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = main(["train", "semantic", "--config", str(config), "--run-id", "cli-smoke"])

    assert exit_code == 0
