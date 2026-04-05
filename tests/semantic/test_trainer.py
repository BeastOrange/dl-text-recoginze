import json
from pathlib import Path

import pytest

from dltr.project import ProjectPaths
from dltr.semantic.config import load_semantic_config
from dltr.semantic.trainer import train_semantic_classifier

torch = pytest.importorskip("torch")


def test_train_semantic_classifier_runs_smoke_epoch(tmp_path: Path) -> None:
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

    config_path = tmp_path / "configs" / "semantic" / "baseline.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: semantic_smoke",
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
                "output_dir: artifacts/checkpoints/semantic/semantic_smoke",
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

    config = load_semantic_config(config_path)
    result = train_semantic_classifier(
        config,
        paths=ProjectPaths.from_root(tmp_path),
        run_id="smoke-run",
    )

    assert result.checkpoint_path.exists()
    assert result.best_checkpoint_path.exists()
    assert result.history_path.exists()
    assert result.history_plot_path.exists()
    assert result.report_path.exists()
