import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from dltr.models.detection.dataset import load_detection_samples
from dltr.models.detection.scaffold import DetectionRunConfig, load_detection_run_config
from dltr.models.detection.trainer import (
    DEFAULT_DETECTION_MODEL_ARCHITECTURE,
    _apply_multi_scale_augmentation,
    _build_detection_model,
    _build_detection_runtime_optimizations,
    _build_train_sampler,
    _prepare_detection_image,
    train_dbnet_detector,
)
from dltr.project import ProjectPaths

torch = pytest.importorskip("torch")


def test_train_dbnet_detector_runs_smoke_epoch(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset_dir = tmp_path / "data" / "raw" / "rects"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    image_a = tmp_path / "det_a.png"
    image_b = tmp_path / "det_b.png"
    _write_box_image(image_a)
    _write_box_image(image_b)

    train_manifest = tmp_path / "data" / "processed" / "detection_splits" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "detection_splits" / "val.jsonl"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "dataset": "rects",
            "image_path": str(image_a),
            "label_path": str(image_a.with_suffix(".json")),
            "instances": [{"points": [4, 4, 24, 4, 24, 24, 4, 24], "text": "营业", "ignore": 0}],
        },
        {
            "dataset": "shopsign",
            "image_path": str(image_b),
            "label_path": str(image_b.with_suffix(".txt")),
            "instances": [{"points": [6, 6, 26, 6, 26, 26, 6, 26], "text": "时间", "ignore": 0}],
        },
    ]
    train_manifest.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
        encoding="utf-8",
    )
    val_manifest.write_text(train_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    config_path = tmp_path / "configs" / "detection" / "dbnet.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: det_smoke",
                "model_name: dbnet",
                "dataset_dir: data/raw/rects",
                "train_manifest: data/processed/detection_splits/train.jsonl",
                "validation_manifest: data/processed/detection_splits/val.jsonl",
                "output_root: artifacts/detection/det_smoke",
                "epochs: 1",
                "batch_size: 2",
                "learning_rate: 0.001",
                "image_height: 32",
                "image_width: 32",
                "device: cpu",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_detection_run_config(config_path)
    result = train_dbnet_detector(
        config,
        paths=ProjectPaths.from_root(tmp_path),
        run_id="smoke-run",
    )

    assert result.checkpoint_path.exists()
    assert result.best_checkpoint_path.exists()
    assert result.history_path.exists()
    assert result.history_markdown_path.exists()
    assert result.history_plot_path.exists()
    assert result.summary_path.exists()
    assert result.report_paths["markdown"].exists()
    history_lines = result.history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1
    assert "train_loss" in history_lines[0]


def test_build_train_sampler_prefers_hard_cases(tmp_path: Path) -> None:
    easy_image = tmp_path / "easy.png"
    hard_image = tmp_path / "hard.png"
    _write_box_image(easy_image)
    _write_box_image(hard_image)
    samples = [
        {
            "dataset": "rects",
            "image_path": str(easy_image),
            "label_path": str(easy_image.with_suffix(".json")),
            "instances": [{"points": [4, 4, 28, 4, 28, 28, 4, 28], "text": "营业", "ignore": 0}],
        },
        {
            "dataset": "rects",
            "image_path": str(hard_image),
            "label_path": str(hard_image.with_suffix(".json")),
            "instances": [
                {"points": [4, 4, 10, 6, 8, 14, 2, 12], "text": "小", "ignore": 0},
                {"points": [12, 4, 18, 6, 16, 14, 10, 12], "text": "字", "ignore": 0},
                {"points": [20, 4, 26, 6, 24, 14, 18, 12], "text": "多", "ignore": 0},
                {"points": [4, 16, 10, 18, 8, 26, 2, 24], "text": "旋", "ignore": 0},
                {"points": [12, 16, 18, 18, 16, 26, 10, 24], "text": "转", "ignore": 0},
            ],
        },
    ]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in samples) + "\n",
        encoding="utf-8",
    )
    loaded_samples = load_detection_samples(manifest)

    sampler = _build_train_sampler(loaded_samples)

    weights = list(sampler.weights.tolist())
    assert weights[1] > weights[0]


def test_apply_multi_scale_augmentation_changes_polygon_scale() -> None:
    image = Image.new("RGB", (64, 64), color="white")
    image_np = __import__("numpy").asarray(image)
    polygons = [[8, 8, 24, 8, 24, 24, 8, 24]]

    augmented_image, augmented_polygons = _apply_multi_scale_augmentation(
        image_np,
        polygons,
        scale_factor=1.5,
        offset_x=8,
        offset_y=6,
    )

    assert augmented_image.shape == image_np.shape
    assert augmented_polygons[0] != polygons[0]




def test_build_detection_model_uses_improved_default_architecture() -> None:
    model = _build_detection_model(torch.nn, architecture=DEFAULT_DETECTION_MODEL_ARCHITECTURE)
    sample = torch.randn(2, 3, 64, 64)

    output = model(sample)

    assert output.shape == (2, 1, 64, 64)


def test_prepare_detection_image_normalizes_rgb_input() -> None:
    image = __import__("numpy").full((16, 16, 3), 255, dtype=__import__("numpy").uint8)

    prepared = _prepare_detection_image(image, target_height=8, target_width=8)

    assert prepared.shape == (8, 8, 3)
    assert prepared.dtype == __import__("numpy").float32
    assert float(prepared.mean()) > 0.0


def test_build_detection_runtime_optimizations_enables_cuda_fastpath() -> None:
    runtime = _build_detection_runtime_optimizations(device="cuda", num_workers=4)

    assert runtime.pin_memory is True
    assert runtime.non_blocking is True
    assert runtime.use_amp is True
    assert runtime.loader_kwargs["pin_memory"] is True
    assert runtime.loader_kwargs["persistent_workers"] is True
    assert runtime.loader_kwargs["prefetch_factor"] == 4


def test_build_detection_runtime_optimizations_keeps_cpu_path_simple() -> None:
    runtime = _build_detection_runtime_optimizations(device="cpu", num_workers=0)

    assert runtime.pin_memory is False
    assert runtime.non_blocking is False
    assert runtime.use_amp is False
    assert runtime.loader_kwargs == {"pin_memory": False}


def test_train_dbnet_detector_resumes_from_checkpoint(tmp_path: Path) -> None:
    config = _build_smoke_detection_config(tmp_path, experiment_name="det_resume_ckpt")
    paths = ProjectPaths.from_root(tmp_path)

    first = train_dbnet_detector(config, paths=paths, run_id="resume-run")
    resumed = train_dbnet_detector(
        replace(config, epochs=2),
        paths=paths,
        resume_from=first.checkpoint_path,
    )

    history_lines = resumed.history_path.read_text(encoding="utf-8").splitlines()
    assert resumed.context.run_dir == first.context.run_dir
    assert len(history_lines) == 2
    assert json.loads(history_lines[-1])["epoch"] == 2


def test_train_dbnet_detector_resumes_from_run_dir(tmp_path: Path) -> None:
    config = _build_smoke_detection_config(tmp_path, experiment_name="det_resume_run_dir")
    paths = ProjectPaths.from_root(tmp_path)

    first = train_dbnet_detector(config, paths=paths, run_id="resume-run-dir")
    resumed = train_dbnet_detector(
        replace(config, epochs=2),
        paths=paths,
        resume_from=first.context.run_dir,
    )

    history_lines = resumed.history_path.read_text(encoding="utf-8").splitlines()
    assert resumed.context.run_dir == first.context.run_dir
    assert len(history_lines) == 2
    assert json.loads(history_lines[-1])["epoch"] == 2


def _write_box_image(path: Path) -> None:
    image = Image.new("RGB", (32, 32), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 24, 24), outline="black", width=2)
    image.save(path)


def _build_smoke_detection_config(tmp_path: Path, *, experiment_name: str) -> DetectionRunConfig:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")
    dataset_dir = tmp_path / "data" / "raw" / "rects"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    image_a = tmp_path / f"{experiment_name}_a.png"
    image_b = tmp_path / f"{experiment_name}_b.png"
    _write_box_image(image_a)
    _write_box_image(image_b)

    train_manifest = tmp_path / "data" / "processed" / "detection_splits" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "detection_splits" / "val.jsonl"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "dataset": "rects",
            "image_path": str(image_a),
            "label_path": str(image_a.with_suffix(".json")),
            "instances": [{"points": [4, 4, 24, 4, 24, 24, 4, 24], "text": "营业", "ignore": 0}],
        },
        {
            "dataset": "shopsign",
            "image_path": str(image_b),
            "label_path": str(image_b.with_suffix(".txt")),
            "instances": [{"points": [6, 6, 26, 6, 26, 26, 6, 26], "text": "时间", "ignore": 0}],
        },
    ]
    content = "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n"
    train_manifest.write_text(content, encoding="utf-8")
    val_manifest.write_text(content, encoding="utf-8")

    config_path = tmp_path / "configs" / "detection" / f"{experiment_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"experiment_name: {experiment_name}",
                "model_name: dbnet",
                "dataset_dir: data/raw/rects",
                "train_manifest: data/processed/detection_splits/train.jsonl",
                "validation_manifest: data/processed/detection_splits/val.jsonl",
                f"output_root: artifacts/detection/{experiment_name}",
                "epochs: 1",
                "batch_size: 2",
                "learning_rate: 0.001",
                "image_height: 32",
                "image_width: 32",
                "device: cpu",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )
    return load_detection_run_config(config_path)
