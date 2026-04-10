"""Visualization helpers for reports."""

from dltr.visualization.eda_markdown import render_eda_markdown
from dltr.visualization.english_benchmark_reports import build_english_benchmark_summary
from dltr.visualization.project_summary import build_project_training_summary
from dltr.visualization.report_index import build_ablation_template, build_training_report_index
from dltr.visualization.training_reports import (
    aggregate_training_runs,
    render_detection_history_plot,
    render_recognition_history_plot,
)

__all__ = [
    "aggregate_training_runs",
    "build_ablation_template",
    "build_english_benchmark_summary",
    "build_project_training_summary",
    "build_training_report_index",
    "render_detection_history_plot",
    "render_eda_markdown",
    "render_recognition_history_plot",
]
