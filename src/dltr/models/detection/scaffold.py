from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from dltr.project import ProjectPaths, discover_project_root


@dataclass(frozen=True)
class DetectionRunConfig:
    experiment_name: str
    model_name: str
    dataset_dir: Path
    annotation_dir: Path | None = None
    output_root: Path | None = None
    train_split: str = "train"
    val_split: str = "val"
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    epochs: int = 120
    batch_size: int = 16
    learning_rate: float = 0.001
    seed: int = 42
    hard_case_sampling: bool = False
    multi_scale_augmentation: bool = False
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must not be empty")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not self.image_extensions:
            raise ValueError("image_extensions must not be empty")


@dataclass(frozen=True)
class DetectionRunContext:
    config: DetectionRunConfig
    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    reports_dir: Path
    exports_dir: Path
    created_at: str


def load_detection_run_config(config_path: str | Path) -> DetectionRunConfig:
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Detection config file not found: {config_file}")

    raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Detection config must be a mapping at the document root")

    root = discover_project_root(config_file.parent)
    data = dict(raw)
    dataset_dir = _resolve_path(data.pop("dataset_dir", None), root, "dataset_dir")
    annotation_dir_raw = data.pop("annotation_dir", None)
    annotation_dir = (
        _resolve_path(annotation_dir_raw, root, "annotation_dir") if annotation_dir_raw else None
    )
    output_root_raw = data.pop("output_root", None)
    output_root = _resolve_path(output_root_raw, root, "output_root") if output_root_raw else None

    config = DetectionRunConfig(
        experiment_name=str(data.pop("experiment_name", "")).strip(),
        model_name=str(data.pop("model_name", "dbnet")).strip(),
        dataset_dir=dataset_dir,
        annotation_dir=annotation_dir,
        output_root=output_root,
        train_split=str(data.pop("train_split", "train")),
        val_split=str(data.pop("val_split", "val")),
        image_extensions=tuple(data.pop("image_extensions", (".jpg", ".jpeg", ".png"))),
        epochs=int(data.pop("epochs", 120)),
        batch_size=int(data.pop("batch_size", 16)),
        learning_rate=float(data.pop("learning_rate", 0.001)),
        seed=int(data.pop("seed", 42)),
        hard_case_sampling=bool(data.pop("hard_case_sampling", False)),
        multi_scale_augmentation=bool(data.pop("multi_scale_augmentation", False)),
        description=str(data.pop("description", "")),
        extra=data,
    )
    config.validate()
    return config


def prepare_detection_run(
    config: DetectionRunConfig,
    *,
    paths: ProjectPaths | None = None,
    run_id: str | None = None,
) -> DetectionRunContext:
    config.validate()
    if not config.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {config.dataset_dir}")
    if config.annotation_dir and not config.annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {config.annotation_dir}")

    project_paths = paths or ProjectPaths.from_root()
    root = config.output_root or (project_paths.artifacts / "detection" / config.experiment_name)
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = root / resolved_run_id
    checkpoints = run_dir / "checkpoints"
    logs = run_dir / "logs"
    reports = run_dir / "reports"
    exports = run_dir / "exports"
    for folder in (run_dir, checkpoints, logs, reports, exports):
        folder.mkdir(parents=True, exist_ok=True)

    return DetectionRunContext(
        config=config,
        run_id=resolved_run_id,
        run_dir=run_dir,
        checkpoints_dir=checkpoints,
        logs_dir=logs,
        reports_dir=reports,
        exports_dir=exports,
        created_at=datetime.now(UTC).isoformat(),
    )


def write_experiment_metadata(
    context: DetectionRunContext,
    *,
    notes: str = "",
) -> dict[str, Path]:
    payload = {
        "run_id": context.run_id,
        "created_at": context.created_at,
        "run_dir": str(context.run_dir),
        "notes": notes,
        "config": _serialize_config(context.config),
    }
    json_path = context.run_dir / "metadata.json"
    md_path = context.run_dir / "metadata.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_metadata_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def write_evaluation_summary(
    context: DetectionRunContext,
    *,
    split: str,
    metrics: dict[str, float],
) -> dict[str, Path]:
    required = {"precision", "recall", "hmean"}
    missing = sorted(required - set(metrics))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required metrics: {missing_text}")

    payload = {
        "run_id": context.run_id,
        "split": split,
        "model_name": context.config.model_name,
        "experiment_name": context.config.experiment_name,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "created_at": datetime.now(UTC).isoformat(),
    }
    json_path = context.reports_dir / f"evaluation_{split}.json"
    md_path = context.reports_dir / f"evaluation_{split}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_evaluation_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def build_export_plan(
    context: DetectionRunContext,
    *,
    checkpoint_path: str | Path,
    targets: tuple[str, ...] = ("onnx", "torchscript"),
) -> dict[str, Path]:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    if checkpoint.suffix not in {".pt", ".pth", ".ckpt"}:
        raise ValueError("Checkpoint extension must be one of: .pt, .pth, .ckpt")
    if not targets:
        raise ValueError("targets must not be empty")

    payload = {
        "run_id": context.run_id,
        "checkpoint": str(checkpoint),
        "targets": list(targets),
        "exports_dir": str(context.exports_dir),
        "commands": [
            (
                "uv run python -m dltr export onnx "
                f"--checkpoint {checkpoint} --output {context.exports_dir / 'model.onnx'}"
            ),
            (
                "uv run python -m dltr export torchscript "
                f"--checkpoint {checkpoint} --output {context.exports_dir / 'model.ts'}"
            ),
        ],
        "created_at": datetime.now(UTC).isoformat(),
    }
    json_path = context.exports_dir / "export_plan.json"
    md_path = context.exports_dir / "export_plan.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_export_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def _resolve_path(value: Any, root: Path, field_name: str) -> Path:
    if value is None:
        raise ValueError(f"{field_name} is required in detection config")
    path = Path(str(value))
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _serialize_config(config: DetectionRunConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("dataset_dir", "annotation_dir", "output_root"):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
    payload["image_extensions"] = list(config.image_extensions)
    return payload


def _metadata_markdown(payload: dict[str, Any]) -> str:
    cfg = payload["config"]
    lines = [
        "# Detection Experiment Metadata",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Created At: `{payload['created_at']}`",
        f"- Experiment: `{cfg['experiment_name']}`",
        f"- Model: `{cfg['model_name']}`",
        f"- Dataset Dir: `{cfg['dataset_dir']}`",
        f"- Notes: {payload['notes'] or 'N/A'}",
        "",
        "## Training Parameters",
        "",
        f"- Epochs: `{cfg['epochs']}`",
        f"- Batch Size: `{cfg['batch_size']}`",
        f"- Learning Rate: `{cfg['learning_rate']}`",
        f"- Hard Case Sampling: `{cfg['hard_case_sampling']}`",
        f"- Multi Scale Augmentation: `{cfg['multi_scale_augmentation']}`",
    ]
    return "\n".join(lines) + "\n"


def _evaluation_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    lines = [
        "# Detection Evaluation Summary",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Split: `{payload['split']}`",
        f"- Experiment: `{payload['experiment_name']}`",
        f"- Model: `{payload['model_name']}`",
        "",
        "## Metrics",
        "",
        f"- Precision: `{metrics['precision']:.6f}`",
        f"- Recall: `{metrics['recall']:.6f}`",
        f"- Hmean: `{metrics['hmean']:.6f}`",
    ]
    return "\n".join(lines) + "\n"


def _export_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Detection Export Plan",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Export Directory: `{payload['exports_dir']}`",
        "",
        "## Targets",
        "",
    ]
    for target in payload["targets"]:
        lines.append(f"- `{target}`")
    lines.extend(
        [
            "",
            "## Suggested Commands",
            "",
        ]
    )
    for command in payload["commands"]:
        lines.append(f"- `{command}`")
    return "\n".join(lines) + "\n"
