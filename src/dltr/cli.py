from __future__ import annotations

import argparse
from collections.abc import Callable

from dltr import __version__
from dltr.commands import (
    cmd_data_build_rec_lmdb,
    cmd_data_prepare_detection,
    cmd_data_prepare_recognition,
    cmd_data_prepare_recognition_crops,
    cmd_data_stats,
    cmd_data_validate,
    cmd_demo,
    cmd_evaluate_detector,
    cmd_evaluate_end2end,
    cmd_evaluate_recognizer,
    cmd_evaluate_semantic,
    cmd_export_onnx,
    cmd_report_summarize_project,
    cmd_report_summarize_training,
    cmd_sync_linux,
    cmd_train_detector,
    cmd_train_recognizer,
    cmd_train_semantic,
)

Handler = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dltr",
        description="Chinese scene text recognition and semantic analysis toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    top = parser.add_subparsers(dest="group")
    top.required = True

    data = top.add_parser("data", help="Dataset validation and analysis commands.")
    data_sub = data.add_subparsers(dest="command")
    data_sub.required = True
    data_validate = data_sub.add_parser("validate")
    data_validate.add_argument("--config", default="configs/data/datasets.example.yaml")
    data_validate.set_defaults(handler=cmd_data_validate)
    data_stats = data_sub.add_parser("stats")
    data_stats.add_argument("--config", default="configs/data/datasets.example.yaml")
    data_stats.add_argument("--output-name", default="dataset_eda_summary.md")
    data_stats.set_defaults(handler=cmd_data_stats)
    data_manifest = data_sub.add_parser("build-rec-lmdb")
    data_manifest.add_argument("--config", default="configs/data/datasets.example.yaml")
    data_manifest.add_argument("--dataset", required=True)
    data_manifest.add_argument("--output")
    data_manifest.set_defaults(handler=cmd_data_build_rec_lmdb)
    data_prepare = data_sub.add_parser("prepare-recognition")
    data_prepare.add_argument("--config", default="configs/data/datasets.example.yaml")
    data_prepare.add_argument("--datasets", nargs="+", required=True)
    data_prepare.add_argument("--combined-output")
    data_prepare.add_argument("--charset-output")
    data_prepare.add_argument("--split-output-dir")
    data_prepare.add_argument("--train-ratio", default=0.8, type=float)
    data_prepare.add_argument("--val-ratio", default=0.1, type=float)
    data_prepare.add_argument("--seed", default=42, type=int)
    data_prepare.add_argument("--min-frequency", default=1, type=int)
    data_prepare.set_defaults(handler=cmd_data_prepare_recognition)
    data_crop_prepare = data_sub.add_parser("prepare-recognition-crops")
    data_crop_prepare.add_argument("--detection-split-dir")
    data_crop_prepare.add_argument("--crop-output-dir")
    data_crop_prepare.add_argument("--recognition-split-dir")
    data_crop_prepare.add_argument("--combined-output")
    data_crop_prepare.add_argument("--charset-output")
    data_crop_prepare.add_argument("--min-frequency", default=1, type=int)
    data_crop_prepare.add_argument("--max-samples", type=int)
    data_crop_prepare.set_defaults(handler=cmd_data_prepare_recognition_crops)
    data_prepare_detection = data_sub.add_parser("prepare-detection")
    data_prepare_detection.add_argument("--config", default="configs/data/datasets.example.yaml")
    data_prepare_detection.add_argument("--datasets", nargs="+", required=True)
    data_prepare_detection.add_argument("--combined-output")
    data_prepare_detection.add_argument("--split-output-dir")
    data_prepare_detection.add_argument("--train-ratio", default=0.8, type=float)
    data_prepare_detection.add_argument("--val-ratio", default=0.1, type=float)
    data_prepare_detection.add_argument("--seed", default=42, type=int)
    data_prepare_detection.set_defaults(handler=cmd_data_prepare_detection)

    train = top.add_parser("train", help="Training commands.")
    train_sub = train.add_subparsers(dest="command")
    train_sub.required = True
    train_detector = train_sub.add_parser("detector")
    train_detector.add_argument("--config", default="configs/detection/dbnet_baseline.yaml")
    train_detector.add_argument("--run-id")
    train_detector.add_argument("--notes", default="")
    train_detector.set_defaults(handler=cmd_train_detector)
    train_recognizer = train_sub.add_parser("recognizer")
    train_recognizer.add_argument(
        "--config",
        default="configs/recognition/transocr_second_pass.yaml",
    )
    train_recognizer.add_argument("--run-id")
    train_recognizer.set_defaults(handler=cmd_train_recognizer)
    train_semantic = train_sub.add_parser("semantic")
    train_semantic.add_argument("--config", default="configs/semantic/macbert_semantic.yaml")
    train_semantic.add_argument("--run-id")
    train_semantic.set_defaults(handler=cmd_train_semantic)

    evaluate = top.add_parser("evaluate", help="Evaluation commands.")
    eval_sub = evaluate.add_subparsers(dest="command")
    eval_sub.required = True
    eval_detector = eval_sub.add_parser("detector")
    eval_detector.add_argument("--config", default="configs/detection/dbnet_baseline.yaml")
    eval_detector.add_argument("--run-id")
    eval_detector.add_argument("--split", default="val")
    eval_detector.add_argument("--precision", required=True, type=float)
    eval_detector.add_argument("--recall", required=True, type=float)
    eval_detector.add_argument("--hmean", required=True, type=float)
    eval_detector.set_defaults(handler=cmd_evaluate_detector)
    eval_recognizer = eval_sub.add_parser("recognizer")
    eval_recognizer.add_argument("--run-name", required=True)
    eval_recognizer.add_argument("--model-name", required=True)
    eval_recognizer.add_argument("--samples", required=True, type=int)
    eval_recognizer.add_argument("--word-accuracy", required=True, type=float)
    eval_recognizer.add_argument("--cer", required=True, type=float)
    eval_recognizer.add_argument("--ned", required=True, type=float)
    eval_recognizer.add_argument("--mean-edit-distance", required=True, type=float)
    eval_recognizer.add_argument("--latency-ms", type=float)
    eval_recognizer.add_argument("--notes")
    eval_recognizer.add_argument("--output-dir")
    eval_recognizer.set_defaults(handler=cmd_evaluate_recognizer)
    eval_semantic = eval_sub.add_parser("semantic")
    eval_semantic.add_argument("--run-name", required=True)
    eval_semantic.add_argument("--predictions-jsonl", required=True)
    eval_semantic.add_argument("--default-class", default="other")
    eval_semantic.add_argument("--output-dir")
    eval_semantic.set_defaults(handler=cmd_evaluate_semantic)
    eval_end2end = eval_sub.add_parser("end2end")
    eval_end2end.add_argument("--text")
    eval_end2end.add_argument("--confidence", type=float)
    eval_end2end.add_argument("--blur-score", type=float)
    eval_end2end.add_argument("--contrast-score", type=float)
    eval_end2end.add_argument("--aspect-ratio", type=float)
    eval_end2end.add_argument("--recognition-config")
    eval_end2end.add_argument("--semantic-class", default="other")
    eval_end2end.add_argument("--output")
    eval_end2end.add_argument("--image")
    eval_end2end.add_argument("--detector-checkpoint")
    eval_end2end.add_argument("--detector-run-dir")
    eval_end2end.add_argument("--recognizer-checkpoint")
    eval_end2end.add_argument("--recognizer-run-dir")
    eval_end2end.add_argument("--output-dir")
    eval_end2end.add_argument("--detector-threshold", default=0.5, type=float)
    eval_end2end.add_argument("--min-area", default=32.0, type=float)
    eval_end2end.set_defaults(handler=cmd_evaluate_end2end)

    export = top.add_parser("export", help="Export commands.")
    export_sub = export.add_subparsers(dest="command")
    export_sub.required = True
    export_onnx = export_sub.add_parser("onnx")
    export_onnx.add_argument("--config", default="configs/detection/dbnet_baseline.yaml")
    export_onnx.add_argument("--checkpoint", required=True)
    export_onnx.add_argument("--run-id")
    export_onnx.set_defaults(handler=cmd_export_onnx)

    demo = top.add_parser("demo", help="Run the English demo app.")
    demo.add_argument("--text", default="营业时间 09:00-21:00 电话 13800138000")
    demo.add_argument("--source-id", default="demo-001")
    demo.add_argument("--semantic-class", default="other")
    demo.add_argument("--confidence", default=0.65, type=float)
    demo.add_argument("--output-dir")
    demo.set_defaults(handler=cmd_demo)

    sync = top.add_parser("sync", help="Synchronize source code to Linux training server.")
    sync_sub = sync.add_subparsers(dest="command")
    sync_sub.required = True
    sync_linux = sync_sub.add_parser("linux")
    sync_linux.add_argument("--remote", required=True)
    sync_linux.add_argument("--target-dir", required=True)
    sync_linux.add_argument("--dry-run", action="store_true")
    sync_linux.set_defaults(handler=cmd_sync_linux)

    report = top.add_parser("report", help="Report and summary commands.")
    report_sub = report.add_subparsers(dest="command")
    report_sub.required = True
    report_training = report_sub.add_parser("summarize-training")
    report_training.add_argument("--task-name", required=True)
    report_training.add_argument("--primary-metric", required=True)
    report_training.add_argument("--run-dirs", nargs="+", required=True)
    report_training.add_argument("--output-dir")
    report_training.set_defaults(handler=cmd_report_summarize_training)
    report_project = report_sub.add_parser("summarize-project")
    report_project.add_argument("--detection-summary-json", required=True)
    report_project.add_argument("--recognition-summary-json", required=True)
    report_project.add_argument("--output-dir")
    report_project.set_defaults(handler=cmd_report_summarize_project)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler = args.handler
    return handler(args)
