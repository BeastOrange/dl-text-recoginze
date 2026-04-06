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

from dltr.data.config import load_data_config
from dltr.data.detection_preparation import (
    build_detection_manifest,
    combine_detection_manifests,
    split_detection_manifest,
)
from dltr.data.hardcase import analyze_hardcase_metadata
from dltr.data.inventory import collect_inventories
from dltr.data.manifest import build_recognition_manifest
from dltr.data.preparation import (
    build_charset_from_manifest,
    combine_recognition_manifests,
    split_manifest,
)
from dltr.data.recognition_crops import extract_recognition_crops_from_detection_manifest
from dltr.data.reporting import write_eda_markdown_report
from dltr.data.semantic_preparation import build_semantic_manifests_from_recognition
from dltr.data.types import DatasetSpec
from dltr.data.validation import validate_dataset_paths
from dltr.models.detection import (
    build_export_plan,
    load_detection_run_config,
    prepare_detection_run,
    train_dbnet_detector,
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
from dltr.models.recognition.trainer import train_crnn_recognizer
from dltr.pipeline.checkpoints import (
    discover_all_run_dirs,
    discover_latest_run_dir,
    resolve_best_checkpoint,
)
from dltr.pipeline.end_to_end import run_end_to_end_pipeline
from dltr.project import ProjectPaths, ensure_runtime_dirs
from dltr.semantic import SemanticPrediction, extract_semantic_slots, generate_semantic_report
from dltr.semantic.classes import SEMANTIC_CLASSES, validate_semantic_class
from dltr.semantic.config import load_semantic_config
from dltr.semantic.trainer import train_semantic_classifier
from dltr.terminal import print_artifact_summary, print_stage_header
from dltr.visualization.ablation_reports import build_ablation_overview
from dltr.visualization.hardcase_reports import build_hardcase_overview
from dltr.visualization.project_summary import build_project_training_summary
from dltr.visualization.report_index import build_ablation_template, build_training_report_index
from dltr.visualization.training_reports import aggregate_training_runs


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
    print_stage_header(
        "开始准备识别数据",
        [
            ("配置文件", _resolve_existing_path_arg(args.config)),
            ("目标数据集", ", ".join(args.datasets)),
        ],
    )
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

    print_artifact_summary(
        "识别数据准备完成，已生成以下产物：",
        [
            ("合并清单", combined_summary.output_path),
            ("字符集文件", charset_summary.output_path),
            ("识别划分目录", split_summary.output_dir),
            ("中文说明摘要", summary_path),
        ],
    )
    return 0


def cmd_data_prepare_recognition_crops(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    print_stage_header("开始根据检测标注裁剪识别样本")
    source_dir = _resolve_output_path(
        args.detection_split_dir,
        paths.data_processed / "detection_splits",
    )
    crop_root = _resolve_output_path(
        args.crop_output_dir,
        paths.data_processed / "recognition_crops",
    )
    split_root = _resolve_output_path(
        args.recognition_split_dir,
        paths.data_processed / "recognition_splits",
    )
    split_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    split_manifests: list[Path] = []
    for split_name in ("train", "val", "test"):
        detection_manifest = source_dir / f"{split_name}.jsonl"
        if not detection_manifest.exists():
            raise FileNotFoundError(f"Detection split manifest not found: {detection_manifest}")
        output_manifest = split_root / f"{split_name}.jsonl"
        summary = extract_recognition_crops_from_detection_manifest(
            split_name=split_name,
            detection_manifest_path=detection_manifest,
            crop_output_dir=crop_root / split_name,
            output_manifest_path=output_manifest,
            max_samples=args.max_samples,
        )
        summaries.append(summary)
        split_manifests.append(output_manifest)

    combined_path = _resolve_output_path(
        args.combined_output,
        paths.data_processed / "recognition_combined.jsonl",
    )
    combine_recognition_manifests(split_manifests, combined_path)
    charset_path = _resolve_output_path(
        args.charset_output,
        paths.data_processed / "charset_zh_mixed.txt",
    )
    build_charset_from_manifest(combined_path, charset_path, min_frequency=args.min_frequency)

    summary_path = paths.data_processed / "recognition_crop_preparation_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Recognition Crop Preparation Summary",
                "",
                f"- Combined Manifest: `{combined_path}`",
                f"- Charset File: `{charset_path}`",
                "",
            ]
            + [
                (
                    f"- `{summary.split_name}`: rows=`{summary.source_rows}`, "
                    f"crops=`{summary.emitted_crops}`, skipped=`{summary.skipped_instances}`"
                )
                for summary in summaries
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print_artifact_summary(
        "识别裁剪数据准备完成，已生成以下产物：",
        [
            ("裁剪图目录", crop_root),
            ("合并清单", combined_path),
            ("字符集文件", charset_path),
            ("识别划分目录", split_root),
            ("中文说明摘要", summary_path),
        ],
    )
    return 0


def cmd_data_prepare_detection(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    config = _load_data_config_arg(args.config, paths)
    print_stage_header(
        "开始准备检测数据",
        [
            ("配置文件", _resolve_existing_path_arg(args.config)),
            ("目标数据集", ", ".join(args.datasets)),
        ],
    )
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
    print_artifact_summary(
        "检测数据准备完成，已生成以下产物：",
        [
            ("合并清单", combined_path),
            ("检测划分目录", split_summary.output_dir),
            ("中文说明摘要", summary_path),
        ],
    )
    return 0


def cmd_data_prepare_semantic(args: argparse.Namespace) -> int:
    paths = ensure_runtime_dirs()
    print_stage_header("开始构建语义数据集")
    outputs = build_semantic_manifests_from_recognition(
        recognition_split_dir=_resolve_output_path(
            args.recognition_split_dir,
            paths.data_processed / "recognition_splits",
        ),
        output_dir=_resolve_output_path(
            args.output_dir,
            paths.root / "data" / "semantic" / "cn_scenetext_sem",
        ),
    )
    print_artifact_summary(
        "语义数据准备完成，已生成以下产物：",
        [
            ("训练集", outputs["train"]),
            ("验证集", outputs["val"]),
            ("测试集", outputs["test"]),
        ],
    )
    return 0


def cmd_train_detector(args: argparse.Namespace) -> int:
    config_path = _resolve_existing_path_arg(args.config)
    config = load_detection_run_config(config_path)
    print_stage_header(
        "开始检测训练",
        [
            ("配置文件", config_path),
            ("实验名称", config.experiment_name),
            ("训练轮数", config.epochs),
            ("批大小", config.batch_size),
            ("图像尺寸", f"{config.image_width}x{config.image_height}"),
        ],
    )
    try:
        result = train_dbnet_detector(
            config,
            paths=ensure_runtime_dirs(),
            run_id=args.run_id,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    except ValueError as exc:
        print(str(exc))
        return 1
    print_artifact_summary(
        "检测训练完成，已生成以下产物：",
        [
            ("运行目录", result.context.run_dir),
            ("最新权重", result.checkpoint_path),
            ("最佳权重", result.best_checkpoint_path),
            ("训练历史", result.history_path),
            ("训练曲线图", result.history_plot_path),
            ("训练摘要", result.summary_path),
            ("评估报告", result.report_paths["markdown"]),
        ],
    )
    return 0


def cmd_train_recognizer(args: argparse.Namespace) -> int:
    config_path = _resolve_existing_path_arg(args.config)
    config = load_recognition_config(config_path)
    print_stage_header(
        "开始识别训练",
        [
            ("配置文件", config_path),
            ("实验名称", config.experiment_name),
            ("训练轮数", config.epochs),
            ("批大小", config.batch_size),
            ("图像尺寸", f"{config.image_width}x{config.image_height}"),
        ],
    )
    if config.model_name != "crnn":
        print(
            "当前仓库还没有接入 TransOCR 的真实训练循环。"
            "请先使用 CRNN 配置训练，或后续再接入 Hugging Face 权重。"
        )
        return 1
    try:
        result = train_crnn_recognizer(
            config,
            paths=ensure_runtime_dirs(),
            run_id=args.run_id,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    print_artifact_summary(
        "识别训练完成，已生成以下产物：",
        [
            ("运行目录", result.run_dir),
            ("最新权重", result.checkpoint_path),
            ("最佳权重", result.best_checkpoint_path),
            ("训练历史", result.history_path),
            ("训练曲线图", result.history_plot_path),
            ("训练摘要", result.summary_path),
            ("评估报告", result.report_path),
        ],
    )
    return 0


def cmd_report_summarize_training(args: argparse.Namespace) -> int:
    run_dirs = [_resolve_existing_path_arg(path) for path in args.run_dirs]
    output_dir = _resolve_output_path(args.output_dir, ProjectPaths.from_root().reports / "train")
    outputs = aggregate_training_runs(
        run_dirs=run_dirs,
        output_dir=output_dir,
        task_name=args.task_name,
        primary_metric=args.primary_metric,
    )
    print(f"json={outputs['json']}")
    print(f"markdown={outputs['markdown']}")
    print(f"png={outputs['png']}")
    return 0


def cmd_report_summarize_project(args: argparse.Namespace) -> int:
    detection_json = _resolve_existing_path_arg(args.detection_summary_json)
    recognition_json = _resolve_existing_path_arg(args.recognition_summary_json)
    semantic_json = (
        _resolve_existing_path_arg(args.semantic_summary_json)
        if args.semantic_summary_json
        else None
    )
    output_dir = _resolve_output_path(args.output_dir, ProjectPaths.from_root().reports / "train")
    outputs = build_project_training_summary(
        detection_summary_json=detection_json,
        recognition_summary_json=recognition_json,
        semantic_summary_json=semantic_json,
        output_dir=output_dir,
    )
    print(f"json={outputs['json']}")
    print(f"markdown={outputs['markdown']}")
    return 0


def cmd_report_build_index(args: argparse.Namespace) -> int:
    train_reports_dir = _resolve_existing_path_arg(args.train_reports_dir)
    output_path = build_training_report_index(
        train_reports_dir=train_reports_dir,
        output_dir=_resolve_output_path(args.output_dir, train_reports_dir),
    )
    print(f"markdown={output_path}")
    return 0


def cmd_report_build_ablation_template(args: argparse.Namespace) -> int:
    output_path = build_ablation_template(
        output_dir=_resolve_output_path(
            args.output_dir,
            ProjectPaths.from_root().reports / "train",
        ),
        task_name=args.task_name,
        experiments=args.experiments,
    )
    print(f"markdown={output_path}")
    return 0


def cmd_report_build_hardcase(args: argparse.Namespace) -> int:
    outputs = build_hardcase_overview(
        config_path=_resolve_existing_path_arg(args.config),
        output_dir=_resolve_output_path(
            args.output_dir,
            ProjectPaths.from_root().reports / "hardcase",
        ),
        project_root=ProjectPaths.from_root().root,
    )
    print(f"markdown={outputs['markdown']}")
    print(f"png={outputs['png']}")
    return 0


def cmd_report_build_ablation_overview(args: argparse.Namespace) -> int:
    outputs = build_ablation_overview(
        detection_summary_json=_resolve_existing_path_arg(args.detection_summary_json),
        recognition_summary_json=_resolve_existing_path_arg(args.recognition_summary_json),
        semantic_summary_json=(
            _resolve_existing_path_arg(args.semantic_summary_json)
            if args.semantic_summary_json
            else None
        ),
        output_dir=_resolve_output_path(
            args.output_dir,
            ProjectPaths.from_root().reports / "ablation",
        ),
    )
    print(f"markdown={outputs['markdown']}")
    print(f"png={outputs['png']}")
    return 0


def cmd_report_build_all(args: argparse.Namespace) -> int:
    root = ProjectPaths.from_root().root
    output_dir = _resolve_output_path(args.output_dir, ProjectPaths.from_root().reports / "train")
    extension_dir = ProjectPaths.from_root().reports / "extensions"
    print_stage_header("开始构建训练汇总报告", [("输出目录", output_dir)])

    detection_root = root / "artifacts" / "detection"
    recognition_root = root / "artifacts" / "checkpoints" / "recognition"
    semantic_root = root / "artifacts" / "checkpoints" / "semantic"

    detection_runs = discover_all_run_dirs(detection_root) if detection_root.exists() else []
    recognition_runs = discover_all_run_dirs(recognition_root) if recognition_root.exists() else []
    semantic_runs = discover_all_run_dirs(semantic_root) if semantic_root.exists() else []

    generated: list[Path] = []
    detection_json: Path | None = None
    recognition_json: Path | None = None
    semantic_json: Path | None = None

    if detection_runs:
        outputs = aggregate_training_runs(
            run_dirs=detection_runs,
            output_dir=output_dir,
            task_name="detection",
            primary_metric="hmean",
        )
        detection_json = outputs["json"]
        generated.extend(outputs.values())
        generated.append(
            build_ablation_template(
                output_dir=output_dir,
                task_name="detection",
                experiments=[run.name for run in detection_runs],
            )
        )

    if recognition_runs:
        outputs = aggregate_training_runs(
            run_dirs=recognition_runs,
            output_dir=output_dir,
            task_name="recognition",
            primary_metric="word_accuracy",
        )
        recognition_json = outputs["json"]
        generated.extend(outputs.values())
        generated.append(
            build_ablation_template(
                output_dir=output_dir,
                task_name="recognition",
                experiments=[run.name for run in recognition_runs],
            )
        )

    if semantic_runs:
        outputs = aggregate_training_runs(
            run_dirs=semantic_runs,
            output_dir=extension_dir,
            task_name="semantic",
            primary_metric="accuracy",
        )
        semantic_json = outputs["json"]
        generated.extend(outputs.values())
        generated.append(
            build_ablation_template(
                output_dir=extension_dir,
                task_name="semantic_extension",
                experiments=[run.name for run in semantic_runs],
            )
        )

    if detection_json and recognition_json:
        project_outputs = build_project_training_summary(
            detection_summary_json=detection_json,
            recognition_summary_json=recognition_json,
            semantic_summary_json=semantic_json,
            output_dir=output_dir,
        )
        generated.extend(project_outputs.values())
        ablation_outputs = build_ablation_overview(
            detection_summary_json=detection_json,
            recognition_summary_json=recognition_json,
            semantic_summary_json=semantic_json,
            output_dir=ProjectPaths.from_root().reports / "ablation",
        )
        generated.extend(ablation_outputs.values())

    generated.append(
        build_training_report_index(
            train_reports_dir=output_dir,
            output_dir=output_dir,
        )
    )
    hardcase_config = ProjectPaths.from_root().configs / "data" / "datasets.example.yaml"
    if hardcase_config.exists():
        hardcase_outputs = build_hardcase_overview(
            config_path=hardcase_config,
            output_dir=ProjectPaths.from_root().reports / "hardcase",
            project_root=ProjectPaths.from_root().root,
        )
        generated.extend(hardcase_outputs.values())
    print_artifact_summary(
        "训练汇总报告构建完成，已生成以下产物：",
        [(f"文件 {index + 1}", path) for index, path in enumerate(generated)],
    )
    return 0


def cmd_train_semantic(args: argparse.Namespace) -> int:
    config_path = _resolve_existing_path_arg(args.config)
    config = load_semantic_config(config_path)
    print_stage_header(
        "开始语义训练",
        [
            ("配置文件", config_path),
            ("实验名称", config.experiment_name),
            ("训练轮数", config.epochs),
            ("批大小", config.batch_size),
            ("类别数", len(config.label_set)),
        ],
    )
    if config.model_name != "char_linear":
        print(
            "当前仓库还没有接入 MacBERT 的真实训练循环。"
            "请先使用 char_linear 配置训练，或后续再接入 Hugging Face 权重。"
        )
        return 1
    try:
        result = train_semantic_classifier(
            config,
            paths=ensure_runtime_dirs(),
            run_id=args.run_id,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    print_artifact_summary(
        "语义训练完成，已生成以下产物：",
        [
            ("运行目录", result.run_dir),
            ("最新权重", result.checkpoint_path),
            ("最佳权重", result.best_checkpoint_path),
            ("训练历史", result.history_path),
            ("训练曲线图", result.history_plot_path),
            ("训练摘要", result.summary_path),
            ("评估报告", result.report_path),
        ],
    )
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
    if args.image:
        detector_checkpoint = _resolve_end_to_end_checkpoint(
            task_name="detection",
            checkpoint=args.detector_checkpoint,
            run_dir=args.detector_run_dir,
        )
        recognizer_checkpoint = _resolve_end_to_end_checkpoint(
            task_name="recognition",
            checkpoint=args.recognizer_checkpoint,
            run_dir=args.recognizer_run_dir,
        )
        artifacts = run_end_to_end_pipeline(
            image_path=_resolve_existing_path_arg(args.image),
            output_dir=_resolve_output_path(
                args.output_dir,
                ProjectPaths.from_root().reports / "eval",
            ),
            detector_checkpoint=detector_checkpoint,
            recognizer_checkpoint=recognizer_checkpoint,
            detector_threshold=args.detector_threshold,
            min_area=args.min_area,
        )
        print(f"json={artifacts.json_path}")
        print(f"markdown={artifacts.markdown_path}")
        print(f"preview={artifacts.preview_image_path}")
        return 0

    if args.text is None or args.confidence is None:
        raise ValueError("Text mode requires --text and --confidence")

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
    if args.serve:
        try:
            app_path = (
                ProjectPaths.from_root().root / "src" / "dltr" / "demo" / "streamlit_app.py"
            ).resolve()
            subprocess.run(
                ["streamlit", "run", str(app_path)],
                cwd=ProjectPaths.from_root().root,
                check=True,
            )
        except ModuleNotFoundError:
            print(
                "Streamlit is not installed in the current environment. "
                "Run `uv sync --extra demo` or use `demo` without `--serve`."
            )
            return 1
        except subprocess.CalledProcessError as exc:
            print(f"Streamlit exited with code {exc.returncode}")
            return exc.returncode
        return 0

    paths = ensure_runtime_dirs()
    prediction = SemanticPrediction(
        source_id=args.source_id,
        text=args.text,
        semantic_class=validate_semantic_class(args.semantic_class),
        confidence=args.confidence,
        slots=extract_semantic_slots(args.text),
    )
    output_dir = _resolve_output_path(args.output_dir, paths.reports / "demo_assets" / "generated")
    report_path = generate_semantic_report(
        run_name="demo_preview",
        predictions=[prediction],
        output_dir=output_dir,
    )
    print("Demo asset generated.")
    print("Static demo assets generated.")
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


def _resolve_end_to_end_checkpoint(
    *,
    task_name: str,
    checkpoint: str | None,
    run_dir: str | None,
) -> Path:
    if checkpoint:
        return _resolve_existing_path_arg(checkpoint)
    if run_dir:
        return resolve_best_checkpoint(_resolve_existing_path_arg(run_dir))
    root = ProjectPaths.from_root().root
    if task_name == "detection":
        default_root = root / "artifacts" / "detection"
    elif task_name == "recognition":
        default_root = root / "artifacts" / "checkpoints" / "recognition"
    else:
        raise ValueError(f"Unsupported task name for checkpoint resolution: {task_name}")

    discovered_run_dir = discover_latest_run_dir(default_root)
    return resolve_best_checkpoint(discovered_run_dir)


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
