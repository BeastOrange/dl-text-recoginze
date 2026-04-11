import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from dltr.models.recognition.config import load_recognition_config
from dltr.models.recognition.metrics import RecognitionScoreSummary
from dltr.models.recognition.trainer import (
    _build_runtime_optimizations,
    train_crnn_recognizer,
    train_transformer_recognizer,
)
from dltr.project import ProjectPaths

torch = pytest.importorskip("torch")


def test_train_crnn_recognizer_runs_smoke_epoch(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")

    train_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "val.jsonl"
    charset_path = tmp_path / "data" / "processed" / "charset_zh_mixed.txt"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    charset_path.parent.mkdir(parents=True, exist_ok=True)

    image_a = tmp_path / "sample_a.png"
    image_b = tmp_path / "sample_b.png"
    _write_text_image(image_a, "营业")
    _write_text_image(image_b, "时间")
    train_manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(image_a),
                        "label_path": str(image_a.with_suffix(".json")),
                        "text": "营业",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "shopsign",
                        "image_path": str(image_b),
                        "label_path": str(image_b.with_suffix(".txt")),
                        "text": "时间",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    val_manifest.write_text(train_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    charset_path.write_text("营\n业\n时\n间\n", encoding="utf-8")

    config_path = tmp_path / "configs" / "recognition" / "crnn.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: crnn_smoke",
                "model_name: crnn",
                "dataset_manifest: data/processed/recognition_splits/train.jsonl",
                "validation_manifest: data/processed/recognition_splits/val.jsonl",
                "charset_file: data/processed/charset_zh_mixed.txt",
                "output_dir: artifacts/checkpoints/recognition/crnn_smoke",
                "epochs: 1",
                "batch_size: 2",
                "image_height: 32",
                "image_width: 128",
                "learning_rate: 0.001",
                "device: cpu",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_recognition_config(config_path)
    result = train_crnn_recognizer(
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
    assert result.report_path.exists()
    assert result.metrics.samples == 2
    history_lines = result.history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1
    assert "train_loss" in history_lines[0]


def test_train_transformer_recognizer_runs_smoke_epoch(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")

    train_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "val.jsonl"
    charset_path = tmp_path / "data" / "processed" / "charset_zh_mixed.txt"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    charset_path.parent.mkdir(parents=True, exist_ok=True)

    image_a = tmp_path / "transformer_a.png"
    image_b = tmp_path / "transformer_b.png"
    _write_text_image(image_a, "营业")
    _write_text_image(image_b, "时间")
    train_manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(image_a),
                        "label_path": str(image_a.with_suffix(".json")),
                        "text": "营业",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "shopsign",
                        "image_path": str(image_b),
                        "label_path": str(image_b.with_suffix(".txt")),
                        "text": "时间",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    val_manifest.write_text(train_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    charset_path.write_text("营\n业\n时\n间\n", encoding="utf-8")

    config_path = tmp_path / "configs" / "recognition" / "transformer.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: transformer_smoke",
                "model_name: transformer",
                "dataset_manifest: data/processed/recognition_splits/train.jsonl",
                "validation_manifest: data/processed/recognition_splits/val.jsonl",
                "charset_file: data/processed/charset_zh_mixed.txt",
                "output_dir: artifacts/checkpoints/recognition/transformer_smoke",
                "epochs: 1",
                "batch_size: 2",
                "image_height: 32",
                "image_width: 128",
                "learning_rate: 0.001",
                "device: cpu",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_recognition_config(config_path)
    result = train_transformer_recognizer(
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
    assert result.report_path.exists()
    assert result.metrics.samples == 2


def test_train_crnn_recognizer_resumes_from_checkpoint_file(tmp_path: Path) -> None:
    config = _build_recognition_smoke_config(
        tmp_path,
        model_name="crnn",
        experiment_name="crnn_resume",
    )
    initial = train_crnn_recognizer(
        config,
        paths=ProjectPaths.from_root(tmp_path),
        run_id="resume-run",
    )
    resumed_config = _build_recognition_smoke_config(
        tmp_path,
        model_name="crnn",
        experiment_name="crnn_resume",
        epochs=2,
    )

    resumed = train_crnn_recognizer(
        resumed_config,
        paths=ProjectPaths.from_root(tmp_path),
        resume_from=initial.checkpoint_path,
    )

    history_lines = resumed.history_path.read_text(encoding="utf-8").splitlines()
    assert resumed.run_dir == initial.run_dir
    assert len(history_lines) == 2
    assert json.loads(history_lines[-1])["epoch"] == 2


def test_train_transformer_recognizer_resumes_from_run_dir(tmp_path: Path) -> None:
    config = _build_recognition_smoke_config(
        tmp_path,
        model_name="transformer",
        experiment_name="transformer_resume",
    )
    initial = train_transformer_recognizer(
        config,
        paths=ProjectPaths.from_root(tmp_path),
        run_id="resume-run",
    )
    resumed_config = _build_recognition_smoke_config(
        tmp_path,
        model_name="transformer",
        experiment_name="transformer_resume",
        epochs=2,
    )

    resumed = train_transformer_recognizer(
        resumed_config,
        paths=ProjectPaths.from_root(tmp_path),
        resume_from=initial.run_dir,
    )

    history_lines = resumed.history_path.read_text(encoding="utf-8").splitlines()
    assert resumed.run_dir == initial.run_dir
    assert len(history_lines) == 2
    assert json.loads(history_lines[-1])["epoch"] == 2


def test_build_runtime_optimizations_enables_cuda_fast_path() -> None:
    runtime = _build_runtime_optimizations(device="cuda", num_workers=8)

    assert runtime.pin_memory is True
    assert runtime.use_amp is True
    assert runtime.loader_kwargs["persistent_workers"] is True
    assert runtime.loader_kwargs["prefetch_factor"] == 4


def test_build_runtime_optimizations_keeps_cpu_path_minimal() -> None:
    runtime = _build_runtime_optimizations(device="cpu", num_workers=0)

    assert runtime.pin_memory is False
    assert runtime.use_amp is False
    assert "persistent_workers" not in runtime.loader_kwargs
    assert "prefetch_factor" not in runtime.loader_kwargs


def test_train_transformer_recognizer_stops_early_when_metric_stagnates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _build_recognition_smoke_config(
        tmp_path,
        model_name="transformer",
        experiment_name="transformer_early_stop",
        epochs=5,
    )
    config = _with_overrides(
        config,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        monitor_metric="word_accuracy",
    )
    summaries = iter(
        [
            RecognitionScoreSummary(2, 0.5, 0.1, 0.1, 0.2),
            RecognitionScoreSummary(2, 0.4, 0.2, 0.2, 0.3),
            RecognitionScoreSummary(2, 0.4, 0.2, 0.2, 0.3),
        ]
    )
    monkeypatch.setattr(
        "dltr.models.recognition.trainer.compute_recognition_scores",
        lambda *_args, **_kwargs: next(summaries),
    )

    result = train_transformer_recognizer(
        config,
        paths=ProjectPaths.from_root(tmp_path),
        run_id="early-stop",
    )

    history = [
        json.loads(line)
        for line in result.history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == 2
    assert history[-1]["epoch"] == 2


def test_train_transformer_recognizer_reduces_learning_rate_on_plateau(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _build_recognition_smoke_config(
        tmp_path,
        model_name="transformer",
        experiment_name="transformer_scheduler",
        epochs=3,
    )
    config = _with_overrides(
        config,
        early_stopping_patience=None,
        monitor_metric="cer",
        lr_scheduler_patience=0,
        lr_scheduler_factor=0.1,
        min_learning_rate=1e-6,
    )
    monkeypatch.setattr(
        "dltr.models.recognition.trainer.compute_recognition_scores",
        lambda *_args, **_kwargs: RecognitionScoreSummary(2, 0.1, 0.5, 0.5, 0.5),
    )

    result = train_transformer_recognizer(
        config,
        paths=ProjectPaths.from_root(tmp_path),
        run_id="scheduler",
    )

    history = [
        json.loads(line)
        for line in result.history_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == 3
    assert history[0]["learning_rate"] == pytest.approx(0.001)
    assert history[-1]["learning_rate"] < history[0]["learning_rate"]


def _write_text_image(path: Path, text: str) -> None:
    image = Image.new("L", (128, 32), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), text, fill=0)
    image.save(path)


def _build_recognition_smoke_config(
    tmp_path: Path,
    *,
    model_name: str,
    experiment_name: str,
    epochs: int = 1,
):
    (tmp_path / "PLAN.md").write_text("plan", encoding="utf-8")

    train_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "train.jsonl"
    val_manifest = tmp_path / "data" / "processed" / "recognition_splits" / "val.jsonl"
    charset_path = tmp_path / "data" / "processed" / "charset_zh_mixed.txt"
    train_manifest.parent.mkdir(parents=True, exist_ok=True)
    charset_path.parent.mkdir(parents=True, exist_ok=True)

    image_a = tmp_path / f"{experiment_name}_a.png"
    image_b = tmp_path / f"{experiment_name}_b.png"
    _write_text_image(image_a, "营业")
    _write_text_image(image_b, "时间")
    train_manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "rects",
                        "image_path": str(image_a),
                        "label_path": str(image_a.with_suffix(".json")),
                        "text": "营业",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "dataset": "shopsign",
                        "image_path": str(image_b),
                        "label_path": str(image_b.with_suffix(".txt")),
                        "text": "时间",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    val_manifest.write_text(train_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    charset_path.write_text("营\n业\n时\n间\n", encoding="utf-8")

    config_path = tmp_path / "configs" / "recognition" / f"{experiment_name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                f"experiment_name: {experiment_name}",
                f"model_name: {model_name}",
                "dataset_manifest: data/processed/recognition_splits/train.jsonl",
                "validation_manifest: data/processed/recognition_splits/val.jsonl",
                "charset_file: data/processed/charset_zh_mixed.txt",
                f"output_dir: artifacts/checkpoints/recognition/{experiment_name}",
                f"epochs: {epochs}",
                "batch_size: 2",
                "image_height: 32",
                "image_width: 128",
                "learning_rate: 0.001",
                "device: cpu",
                "num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )
    return load_recognition_config(config_path)


def _with_overrides(config, **overrides):  # noqa: ANN001, ANN202
    payload = config.__dict__ | overrides
    return type(config)(**payload)
