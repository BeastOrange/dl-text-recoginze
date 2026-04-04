from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from dltr.data import (
    analyze_hardcase_metadata,
    build_charset_from_manifest,
    build_detection_manifest,
    build_recognition_manifest,
    collect_inventories,
    combine_detection_manifests,
    combine_recognition_manifests,
    load_data_config,
    split_detection_manifest,
    split_manifest,
    validate_dataset_paths,
    write_eda_markdown_report,
)
from dltr.data.types import DatasetSpec
from dltr.models.detection import (
    build_export_plan,
    load_detection_run_config,
    prepare_detection_run,
    write_evaluation_summary,
    write_experiment_metadata,
)
from dltr.models.recognition.config import (
    RecognitionExperimentConfig,
    SecondPassConfig,
    load_recognition_config,
)
from dltr.models.recognition.evaluation import (
    RecognitionMetrics,
    generate_recognition_evaluation_report,
)
from dltr.models.recognition.refinement import (
    QualitySignals,
    second_pass_reasons,
    should_apply_second_pass,
)
from dltr.project import ProjectPaths, ensure_runtime_dirs
from dltr.semantic import SemanticPrediction, extract_semantic_slots, generate_semantic_report
from dltr.semantic.classes import SEMANTIC_CLASSES, validate_semantic_class


@dataclass(frozen=True)
class SemanticTrainConfig:
    experiment_name: str
    model_name: str
    label_set: list[str]
    dataset_manifest: str
    validation_manifest: str
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    max_length: int

    def validate(self) -> None:
        if not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not self.label_set:
            raise ValueError("label_set must not be empty")
        invalid = sorted({label for label in self.label_set if label not in SEMANTIC_CLASSES})
        if invalid:
            raise ValueError(f"Unsupported semantic labels: {', '.join(invalid)}")
        if not self.dataset_manifest.strip():
            raise ValueError("dataset_manifest must be non-empty")
        if not self.validation_manifest.strip():
            raise ValueError("validation_manifest must be non-empty")
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")


def cmd_data_validate(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    config = _load_data_config_arg(args.config, paths)
    summary = validate_dataset_paths(paths, config)
    print("Dataset validation summary:")
    for item in summary.dataset_results:
        status = "OK" if not item.issues else "ISSUES"
        print(f"- {item.name}: {status}")
        print(f"  resolved_path: {item.resolved_path}")
        if item.issues:
            for issue in item.issues:
                print(f"  issue: {issue}")
    return 0 if summary.ok else 1


def cmd_data_stats(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    config = _load_data_config_arg(args.config, paths)
    validation = validate_dataset_paths(paths, config)
    inventories = collect_inventories(paths, config)
    hardcases = {name: analyze_hardcase_metadata(inv) for name, inv in inventories.items()}
    report_path = write_eda_markdown_report(
        paths,
        config,
        validation,
        inventories,
        hardcases,
        filename=args.output_name,
    )
    print(f"EDA report written to {report_path}")
    return 0


def cmd_data_build_rec_lmdb(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    config = _load_data_config_arg(args.config, paths)
    spec = _find_dataset_spec(config.datasets, args.dataset)
    output_path = _resolve_output_path(
        args.output,
        paths.data_processed / f"{spec.name}_manifest.jsonl",
    )
    result = build_recognition_manifest(
        dataset_name=spec.name,
        dataset_root=paths.root / spec.relative_path,
        output_path=output_path,
        image_extensions=spec.image_extensions,
        label_extensions=spec.label_extensions,
    )
    print("Manifest scaffold completed.")
    print("Note: current scaffold emits JSONL manifest, not LMDB.")
    print(
        f"dataset={result.dataset_name} scanned={result.scanned_images} "
        f"emitted={result.emitted_rows} skipped={result.skipped_without_label}"
    )
    print(f"output={result.output_path}")
    return 0


def cmd_data_prepare_recognition(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    config = _load_data_config_arg(args.config, paths)
    selected_specs = [_find_dataset_spec(config.datasets, name) for name in args.datasets]

    manifest_paths: list[Path] = []
    manifest_dir = paths.data_processed / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for spec in selected_specs:
        manifest_path = manifest_dir / f"{spec.name}.jsonl"
        build_recognition_manifest(
            dataset_name=spec.name,
            dataset_root=paths.root / spec.relative_path,
            output_path=manifest_path,
            image_extensions=spec.image_extensions,
            label_extensions=spec.label_extensions,
        )
        manifest_paths.append(manifest_path)

    combined_path = _resolve_output_path(
        args.combined_output,
        paths.data_processed / "recognition_combined.jsonl",
    )
    combined_summary = combine_recognition_manifests(
        manifest_paths=manifest_paths,
        output_path=combined_path,
    )

    charset_path = _resolve_output_path(
        args.charset_output,
        paths.data_processed / "charset_zh_mixed.txt",
    )
    charset_summary = build_charset_from_manifest(
        combined_path,
        charset_path,
        min_frequency=args.min_frequency,
    )

    split_dir = _resolve_output_path(
        args.split_output_dir,
        paths.data_processed / "recognition_splits",
    )
    split_summary = split_manifest(
        combined_path,
        split_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    summary_path = paths.data_processed / "recognition_preparation_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Recognition Preparation Summary",
                "",
                f"- Combined Manifest: `{combined_summary.output_path}`",
                f"- Total Rows: `{combined_summary.total_rows}`",
                f"- Charset File: `{charset_summary.output_path}`",
                f"- Unique Characters: `{charset_summary.unique_characters}`",
                f"- Train Rows: `{split_summary.train_rows}`",
                f"- Val Rows: `{split_summary.val_rows}`",
                f"- Test Rows: `{split_summary.test_rows}`",
                "",
                "## Dataset Counts",
                "",
            ]
            + [
                f"- `{dataset}`: `{count}`"
                for dataset, count in sorted(combined_summary.dataset_counts.items())
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"combined_manifest={combined_summary.output_path}")
    print(f"charset={charset_summary.output_path}")
    print(f"split_dir={split_summary.output_dir}")
    print(f"summary={summary_path}")
    return 0


def cmd_data_prepare_detection(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    config = _load_data_config_arg(args.config, paths)
    selected_specs = [_find_dataset_spec(config.datasets, name) for name in args.datasets]

    manifest_paths: list[Path] = []
    manifest_dir = paths.data_processed / "detection_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for spec in selected_specs:
        manifest_path = manifest_dir / f"{spec.name}.jsonl"
        build_detection_manifest(
            dataset_name=spec.name,
            dataset_root=paths.root / spec.relative_path,
            output_path=manifest_path,
            image_extensions=spec.image_extensions,
            label_extensions=spec.label_extensions,
        )
        manifest_paths.append(manifest_path)

    combined_path = _resolve_output_path(
        args.combined_output,
        paths.data_processed / "detection_combined.jsonl",
    )
    dataset_counts = combine_detection_manifests(
        manifest_paths=manifest_paths,
        output_path=combined_path,
    )
    split_dir = _resolve_output_path(
        args.split_output_dir,
        paths.data_processed / "detection_splits",
    )
    split_summary = split_detection_manifest(
        combined_path,
        split_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    summary_path = paths.data_processed / "detection_preparation_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Detection Preparation Summary",
                "",
                f"- Combined Manifest: `{combined_path}`",
                f"- Train Rows: `{split_summary.train_rows}`",
                f"- Val Rows: `{split_summary.val_rows}`",
                f"- Test Rows: `{split_summary.test_rows}`",
                "",
                "## Dataset Counts",
                "",
            ]
            + [f"- `{dataset}`: `{count}`" for dataset, count in sorted(dataset_counts.items())]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"combined_manifest={combined_path}")
    print(f"split_dir={split_summary.output_dir}")
    print(f"summary={summary_path}")
    return 0


def cmd_train_detector(args: argparse.Namespace) -> int:
    config = load_detection_run_config(_resolve_existing_path_arg(args.config))
    context = prepare_detection_run(config, run_id=args.run_id)
    artifacts = write_experiment_metadata(context, notes=args.notes)
    print(f"Detection run scaffold created: {context.run_dir}")
    print(f"metadata_json={artifacts['json']}")
    print(f"metadata_markdown={artifacts['markdown']}")
    return 0


def cmd_train_recognizer(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    config = load_recognition_config(_resolve_existing_path_arg(args.config))
    run_dir = _prepare_recognition_run(paths, config, args.run_id)
    outputs = _write_recognition_train_plan(run_dir, config)
    print(f"Recognition run scaffold created: {run_dir}")
    print(f"plan_json={outputs['json']}")
    print(f"plan_markdown={outputs['markdown']}")
    return 0


def cmd_train_semantic(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    config = _load_semantic_train_config(_resolve_existing_path_arg(args.config))
    run_dir = _prepare_semantic_run(paths, config, args.run_id)
    outputs = _write_semantic_train_plan(run_dir, config)
    print(f"Semantic run scaffold created: {run_dir}")
    print(f"plan_json={outputs['json']}")
    print(f"plan_markdown={outputs['markdown']}")
    return 0


def cmd_evaluate_detector(args: argparse.Namespace) -> int:
    config = load_detection_run_config(_resolve_existing_path_arg(args.config))
    context = prepare_detection_run(config, run_id=args.run_id)
    write_experiment_metadata(
        context,
        notes="Created during detection evaluation summary generation.",
    )
    outputs = write_evaluation_summary(
        context,
        split=args.split,
        metrics={
            "precision": args.precision,
            "recall": args.recall,
            "hmean": args.hmean,
        },
    )
    print(f"Detection evaluation summary written to {outputs['markdown']}")
    return 0


def cmd_evaluate_recognizer(args: argparse.Namespace) -> int:
    metrics = RecognitionMetrics(
        samples=args.samples,
        word_accuracy=args.word_accuracy,
        cer=args.cer,
        ned=args.ned,
        mean_edit_distance=args.mean_edit_distance,
        latency_ms=args.latency_ms,
    )
    output_dir = _resolve_output_path(args.output_dir, ProjectPaths.from_root().reports / "eval")
    report_path = generate_recognition_evaluation_report(
        run_name=args.run_name,
        model_name=args.model_name,
        metrics=metrics,
        output_dir=output_dir,
        notes=args.notes,
    )
    print(f"Recognition evaluation report written to {report_path}")
    return 0


def cmd_evaluate_semantic(args: argparse.Namespace) -> int:
    predictions_path = _resolve_existing_path_arg(args.predictions_jsonl)
    predictions = _load_semantic_predictions(predictions_path, args.default_class)
    output_dir = _resolve_output_path(args.output_dir, ProjectPaths.from_root().reports / "eval")
    report_path = generate_semantic_report(
        run_name=args.run_name,
        predictions=predictions,
        output_dir=output_dir,
    )
    print(f"Semantic evaluation report written to {report_path}")
    return 0


def cmd_evaluate_end2end(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    policy = _load_second_pass_policy(args.recognition_config)
    quality = QualitySignals(
        blur_score=args.blur_score,
        contrast_score=args.contrast_score,
        aspect_ratio=args.aspect_ratio,
    )
    reasons = second_pass_reasons(args.confidence, args.text, quality, policy)
    slots = extract_semantic_slots(args.text)
    output_path = _resolve_output_path(
        args.output,
        paths.reports / "eval" / "end2end_preview.json",
    )
    payload = {
        "text": args.text,
        "confidence": args.confidence,
        "second_pass_enabled": policy.enabled,
        "should_apply_second_pass": should_apply_second_pass(
            args.confidence,
            args.text,
            quality,
            policy,
        ),
        "reasons": reasons,
        "quality": asdict(quality),
        "semantic_class": validate_semantic_class(args.semantic_class),
        "slots": asdict(slots),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"End-to-end preview written to {output_path}")
    return 0


def cmd_export_onnx(args: argparse.Namespace) -> int:
    config = load_detection_run_config(_resolve_existing_path_arg(args.config))
    context = prepare_detection_run(config, run_id=args.run_id)
    outputs = build_export_plan(
        context,
        checkpoint_path=_resolve_existing_path_arg(args.checkpoint),
        targets=("onnx",),
    )
    print("Export plan created for ONNX conversion.")
    print(f"plan_json={outputs['json']}")
    print(f"plan_markdown={outputs['markdown']}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    prediction = SemanticPrediction(
        source_id=args.source_id,
        text=args.text,
        semantic_class=validate_semantic_class(args.semantic_class),
        confidence=args.confidence,
        slots=extract_semantic_slots(args.text),
    )
    output_dir = _resolve_output_path(args.output_dir, paths.reports / "demo_assets")
    report_path = generate_semantic_report(
        run_name="demo_preview",
        predictions=[prediction],
        output_dir=output_dir,
    )
    print("Demo asset generated.")
    print(
        "Note: Streamlit UI is planned later; this command currently generates "
        "English demo assets."
    )
    print(f"report={report_path}")
    return 0


def cmd_sync_linux(args: argparse.Namespace) -> int:
    paths = ProjectPaths.from_root()
    script = paths.scripts / "sync_to_linux.sh"
    if not script.exists():
        raise FileNotFoundError(f"Sync script not found: {script}")
    env = {
        **os.environ,
        "REMOTE": args.remote,
        "TARGET_DIR": args.target_dir,
        "DRY_RUN": "1" if args.dry_run else "0",
    }
    result = subprocess.run(["bash", str(script)], cwd=paths.root, env=env, check=False)
    return int(result.returncode)


def _load_data_config_arg(config_arg: str | None, paths: ProjectPaths) -> Any:
    config_path = _resolve_existing_path_arg(
        config_arg or paths.configs / "data" / "datasets.example.yaml",
    )
    return load_data_config(config_path)


def _resolve_existing_path_arg(value: str | Path) -> Path:
    path = Path(value)
    resolved = (
        path.resolve()
        if path.is_absolute()
        else (ProjectPaths.from_root().root / path).resolve()
    )
    if not resolved.exists():
        raise FileNotFoundError(f"Required path not found: {resolved}")
    return resolved


def _resolve_output_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (ProjectPaths.from_root().root / raw).resolve()


def _find_dataset_spec(specs: list[DatasetSpec], name: str) -> DatasetSpec:
    for spec in specs:
        if spec.name == name:
            return spec
    available = ", ".join(sorted(spec.name for spec in specs))
    raise ValueError(f"Unknown dataset '{name}'. Available: {available}")


def _prepare_recognition_run(
    paths: ProjectPaths,
    config: RecognitionExperimentConfig,
    run_id: str | None,
) -> Path:
    dataset_manifest = (paths.root / config.dataset_manifest).resolve()
    charset_file = (paths.root / config.charset_file).resolve()
    if not dataset_manifest.exists():
        raise FileNotFoundError(f"Recognition manifest not found: {dataset_manifest}")
    if not charset_file.exists():
        raise FileNotFoundError(f"Charset file not found: {charset_file}")

    output_root = (paths.root / config.output_dir).resolve()
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_recognition_train_plan(
    run_dir: Path,
    config: RecognitionExperimentConfig,
) -> dict[str, Path]:
    payload = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "dataset_manifest": config.dataset_manifest,
        "charset_file": config.charset_file,
        "output_dir": str(run_dir),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "image_height": config.image_height,
        "image_width": config.image_width,
        "learning_rate": config.learning_rate,
        "second_pass": asdict(config.second_pass),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    json_path = run_dir / "train_plan.json"
    markdown_path = run_dir / "train_plan.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                f"# Recognition Training Plan: {config.experiment_name}",
                "",
                f"- Model: `{config.model_name}`",
                f"- Dataset Manifest: `{config.dataset_manifest}`",
                f"- Charset File: `{config.charset_file}`",
                f"- Epochs: `{config.epochs}`",
                f"- Batch Size: `{config.batch_size}`",
                f"- Learning Rate: `{config.learning_rate}`",
                f"- Second Pass Enabled: `{config.second_pass.enabled}`",
                f"- Confidence Threshold: `{config.second_pass.confidence_threshold}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}


def _load_semantic_train_config(config_path: Path) -> SemanticTrainConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Semantic config must be a YAML mapping")
    config = SemanticTrainConfig(
        experiment_name=str(payload.get("experiment_name", "")).strip(),
        model_name=str(payload.get("model_name", "")).strip(),
        label_set=[str(item).strip() for item in payload.get("label_set", [])],
        dataset_manifest=str(payload.get("dataset_manifest", "")).strip(),
        validation_manifest=str(payload.get("validation_manifest", "")).strip(),
        output_dir=str(payload.get("output_dir", "")).strip(),
        epochs=int(payload.get("epochs", 0)),
        batch_size=int(payload.get("batch_size", 0)),
        learning_rate=float(payload.get("learning_rate", 0.0)),
        max_length=int(payload.get("max_length", 0)),
    )
    config.validate()
    return config


def _prepare_semantic_run(
    paths: ProjectPaths,
    config: SemanticTrainConfig,
    run_id: str | None,
) -> Path:
    dataset_manifest = (paths.root / config.dataset_manifest).resolve()
    validation_manifest = (paths.root / config.validation_manifest).resolve()
    if not dataset_manifest.exists():
        raise FileNotFoundError(f"Semantic dataset manifest not found: {dataset_manifest}")
    if not validation_manifest.exists():
        raise FileNotFoundError(f"Semantic validation manifest not found: {validation_manifest}")

    output_root = (paths.root / config.output_dir).resolve()
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_semantic_train_plan(
    run_dir: Path,
    config: SemanticTrainConfig,
) -> dict[str, Path]:
    payload = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "label_set": config.label_set,
        "dataset_manifest": config.dataset_manifest,
        "validation_manifest": config.validation_manifest,
        "output_dir": str(run_dir),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    json_path = run_dir / "train_plan.json"
    markdown_path = run_dir / "train_plan.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                f"# Semantic Training Plan: {config.experiment_name}",
                "",
                f"- Model: `{config.model_name}`",
                f"- Labels: `{', '.join(config.label_set)}`",
                f"- Train Manifest: `{config.dataset_manifest}`",
                f"- Validation Manifest: `{config.validation_manifest}`",
                f"- Epochs: `{config.epochs}`",
                f"- Batch Size: `{config.batch_size}`",
                f"- Learning Rate: `{config.learning_rate}`",
                f"- Max Length: `{config.max_length}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}


def _load_semantic_predictions(
    predictions_path: Path,
    default_class: str,
) -> list[SemanticPrediction]:
    resolved_default_class = validate_semantic_class(default_class)
    predictions: list[SemanticPrediction] = []
    for line_number, raw_line in enumerate(
        predictions_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError(f"Line {line_number} in predictions file is missing text")
        semantic_class = str(payload.get("semantic_class", resolved_default_class)).strip()
        slots = extract_semantic_slots(text)
        predictions.append(
            SemanticPrediction(
                source_id=str(payload.get("source_id", f"line-{line_number}")).strip(),
                text=text,
                semantic_class=validate_semantic_class(semantic_class),
                confidence=float(payload.get("confidence", 0.5)),
                slots=slots,
            )
        )
    if not predictions:
        raise ValueError(f"No valid predictions found in {predictions_path}")
    return predictions


def _load_second_pass_policy(config_path: str | None) -> SecondPassConfig:
    if config_path:
        return load_recognition_config(_resolve_existing_path_arg(config_path)).second_pass
    return SecondPassConfig()
