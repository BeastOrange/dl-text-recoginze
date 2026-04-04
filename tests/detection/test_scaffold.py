from __future__ import annotations

import json
from pathlib import Path

import pytest

from dltr.models.detection.scaffold import (
    DetectionRunConfig,
    build_export_plan,
    load_detection_run_config,
    prepare_detection_run,
    write_evaluation_summary,
    write_experiment_metadata,
)
from dltr.project import ProjectPaths


def test_load_detection_run_config_parses_and_resolves_paths(tmp_path: Path) -> None:
    root = tmp_path
    (root / "PLAN.md").write_text("plan", encoding="utf-8")
    data_dir = root / "data" / "raw" / "rctw17" / "images"
    data_dir.mkdir(parents=True)
    config_dir = root / "configs" / "detection"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "baseline.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: det_baseline",
                "model_name: dbnet",
                "dataset_dir: data/raw/rctw17/images",
                "epochs: 100",
                "batch_size: 8",
                "learning_rate: 0.001",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_detection_run_config(config_path)

    assert isinstance(cfg, DetectionRunConfig)
    assert cfg.experiment_name == "det_baseline"
    assert cfg.dataset_dir == data_dir
    assert cfg.epochs == 100


def test_prepare_detection_run_raises_when_dataset_missing(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    paths = ProjectPaths.from_root(tmp_path)
    cfg = DetectionRunConfig(
        experiment_name="det_missing",
        model_name="dbnet",
        dataset_dir=tmp_path / "data" / "raw" / "not-exist",
    )

    with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
        prepare_detection_run(cfg, paths=paths, run_id="run-001")


def test_prepare_detection_run_creates_expected_layout(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset = tmp_path / "data" / "raw" / "rctw17" / "images"
    dataset.mkdir(parents=True)
    paths = ProjectPaths.from_root(tmp_path)
    cfg = DetectionRunConfig(
        experiment_name="det_layout",
        model_name="dbnet",
        dataset_dir=dataset,
    )

    run = prepare_detection_run(cfg, paths=paths, run_id="run-002")

    assert run.run_dir.exists()
    assert run.checkpoints_dir.exists()
    assert run.logs_dir.exists()
    assert run.reports_dir.exists()
    assert run.exports_dir.exists()


def test_write_metadata_and_evaluation_outputs(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset = tmp_path / "data" / "raw" / "rctw17" / "images"
    dataset.mkdir(parents=True)
    paths = ProjectPaths.from_root(tmp_path)
    cfg = DetectionRunConfig(
        experiment_name="det_report",
        model_name="dbnet",
        dataset_dir=dataset,
    )
    run = prepare_detection_run(cfg, paths=paths, run_id="run-003")

    metadata_paths = write_experiment_metadata(run, notes="Baseline experiment")
    eval_paths = write_evaluation_summary(
        run,
        split="val",
        metrics={"precision": 0.81, "recall": 0.79, "hmean": 0.80},
    )

    assert metadata_paths["json"].exists()
    assert metadata_paths["markdown"].exists()
    assert eval_paths["json"].exists()
    assert eval_paths["markdown"].exists()

    payload = json.loads(metadata_paths["json"].read_text(encoding="utf-8"))
    assert payload["config"]["experiment_name"] == "det_report"
    assert payload["notes"] == "Baseline experiment"

    eval_payload = json.loads(eval_paths["json"].read_text(encoding="utf-8"))
    assert eval_payload["split"] == "val"
    assert eval_payload["metrics"]["hmean"] == 0.80


def test_build_export_plan_fails_on_missing_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset = tmp_path / "data" / "raw" / "rctw17" / "images"
    dataset.mkdir(parents=True)
    paths = ProjectPaths.from_root(tmp_path)
    cfg = DetectionRunConfig(
        experiment_name="det_export_missing",
        model_name="dbnet",
        dataset_dir=dataset,
    )
    run = prepare_detection_run(cfg, paths=paths, run_id="run-004")

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        build_export_plan(run, checkpoint_path=run.checkpoints_dir / "best.pth")


def test_build_export_plan_writes_json_and_markdown(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset = tmp_path / "data" / "raw" / "rctw17" / "images"
    dataset.mkdir(parents=True)
    paths = ProjectPaths.from_root(tmp_path)
    cfg = DetectionRunConfig(
        experiment_name="det_export",
        model_name="dbnet",
        dataset_dir=dataset,
    )
    run = prepare_detection_run(cfg, paths=paths, run_id="run-005")
    checkpoint = run.checkpoints_dir / "best.pth"
    checkpoint.write_text("placeholder", encoding="utf-8")

    outputs = build_export_plan(run, checkpoint_path=checkpoint)

    assert outputs["json"].exists()
    assert outputs["markdown"].exists()
    content = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert content["checkpoint"] == str(checkpoint)
    assert "onnx" in content["targets"]
