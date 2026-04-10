from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from dltr.visualization.plot_style import (
    bar_colors,
    format_compact_label,
    resolve_label_rotation,
    resolve_summary_fig_width,
    resolve_upper_bound,
    style_axis,
)

OBSOLETE_RUN_MARKERS = {"report-smoke"}
OBSOLETE_EXPERIMENT_MARKERS = {"report_smoke"}
LOSS_COLOR = "#2F6CAD"
METRIC_COLOR = "#4A8F6D"
ALT_METRIC_COLOR = "#D26D3D"
BAR_EDGE_COLOR = "#2E3A46"


def render_recognition_history_plot(
    *,
    run_name: str,
    history_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    history = _load_history(history_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{run_name}_training_curve.png"
    markdown_path = output_dir / f"{run_name}_training_curve.md"
    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    word_accuracy = [float(item["val_word_accuracy"]) for item in history]
    cer = [float(item["val_cer"]) for item in history]
    ned = [float(item["val_ned"]) for item in history]
    _render_dual_axis_history_plot(
        png_path=png_path,
        epochs=epochs,
        left_title="Recognition Loss / CER",
        right_title="Recognition Accuracy / NED",
        left_series=[
            (train_loss, "Train Loss", LOSS_COLOR, "o"),
            (cer, "Val CER", ALT_METRIC_COLOR, "s"),
        ],
        right_series=[
            (word_accuracy, "Val Word Accuracy", METRIC_COLOR, "o"),
            (ned, "Val NED", "#7E63B8", "s"),
        ],
    )
    markdown_path.write_text(
        "\n".join(
            [
                f"# Recognition Training Curves: {run_name}",
                "",
                f"- Image: `{png_path.name}`",
                f"- Epochs: `{len(history)}`",
                f"- Best Word Accuracy: `{max(word_accuracy):.6f}`",
                f"- Lowest CER: `{min(cer):.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"png": png_path, "markdown": markdown_path}


def render_detection_history_plot(
    *,
    run_name: str,
    history_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    history = _load_history(history_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{run_name}_training_curve.png"
    markdown_path = output_dir / f"{run_name}_training_curve.md"
    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    precision = [float(item["val_precision"]) for item in history]
    recall = [float(item["val_recall"]) for item in history]
    hmean = [float(item["val_hmean"]) for item in history]
    _render_dual_axis_history_plot(
        png_path=png_path,
        epochs=epochs,
        left_title="Detection Loss / Hmean",
        right_title="Detection Precision / Recall",
        left_series=[
            (train_loss, "Train Loss", LOSS_COLOR, "o"),
            (hmean, "Val Hmean", METRIC_COLOR, "s"),
        ],
        right_series=[
            (precision, "Val Precision", "#2E84A6", "o"),
            (recall, "Val Recall", "#C76056", "s"),
        ],
    )
    markdown_path.write_text(
        "\n".join(
            [
                f"# Detection Training Curves: {run_name}",
                "",
                f"- Image: `{png_path.name}`",
                f"- Epochs: `{len(history)}`",
                f"- Best Hmean: `{max(hmean):.6f}`",
                f"- Best Precision: `{max(precision):.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"png": png_path, "markdown": markdown_path}


def aggregate_training_runs(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    task_name: str,
    primary_metric: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for run_dir in run_dirs:
        summary_path = run_dir / "training_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if _is_obsolete_run(run_dir=run_dir, payload=payload):
            continue
        metrics = payload.get("metrics", {})
        records.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "primary_metric": float(metrics.get(primary_metric, 0.0)),
                "metrics": metrics,
                "best_checkpoint_path": payload.get("best_checkpoint_path"),
            }
        )
    records.sort(key=lambda item: item["primary_metric"], reverse=True)
    json_path = output_dir / f"{task_name}_summary.json"
    markdown_path = output_dir / f"{task_name}_summary.md"
    png_path = output_dir / f"{task_name}_summary.png"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                f"# {task_name.title()} Training Summary",
                "",
                f"- Primary Metric: `{primary_metric}`",
                "",
                "| Run | Primary Metric | Best Checkpoint |",
                "|---|---:|---|",
            ]
            + [
                f"| {item['run_name']} | {item['primary_metric']:.6f} | "
                f"{item['best_checkpoint_path'] or '-'} |"
                for item in records
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _render_summary_bar_plot(
        png_path=png_path,
        run_names=[item["run_name"] for item in records],
        primary_values=[item["primary_metric"] for item in records],
        task_name=task_name,
        primary_metric=primary_metric,
    )
    return {"json": json_path, "markdown": markdown_path, "png": png_path}


def _render_dual_axis_history_plot(
    *,
    png_path: Path,
    epochs: list[int],
    left_title: str,
    right_title: str,
    left_series: list[tuple[list[float], str, str, str]],
    right_series: list[tuple[list[float], str, str, str]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    _draw_axis_series(axes[0], epochs, left_title, left_series)
    _draw_axis_series(axes[1], epochs, right_title, right_series)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def _draw_axis_series(
    axis: plt.Axes,
    epochs: list[int],
    title: str,
    series: list[tuple[list[float], str, str, str]],
) -> None:
    style_axis(axis)
    for values, label, color, marker in series:
        axis.plot(
            epochs,
            values,
            marker=marker,
            markersize=4.8,
            linewidth=1.9,
            color=color,
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Value")
    axis.legend()


def _render_summary_bar_plot(
    *,
    png_path: Path,
    run_names: list[str],
    primary_values: list[float],
    task_name: str,
    primary_metric: str,
) -> None:
    fig, ax = plt.subplots(figsize=(resolve_summary_fig_width(run_names), 5.4))
    style_axis(ax)
    x_positions = list(range(len(run_names)))
    labels = [format_compact_label(name) for name in run_names]
    rotation = resolve_label_rotation(run_names)
    if run_names:
        bars = ax.bar(
            x_positions,
            primary_values,
            color=bar_colors(len(run_names)),
            width=0.68,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.6,
        )
        upper_bound = resolve_upper_bound(max(primary_values))
        ax.set_ylim(0.0, upper_bound)
        for bar, value in zip(bars, primary_values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper_bound * 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#2E3A46",
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    else:
        ax.set_xticks([])
        ax.set_ylim(0.0, 1.0)
        ax.text(
            0.5,
            0.5,
            "No available runs",
            ha="center",
            va="center",
            fontsize=11,
            color="#6B7B8D",
            transform=ax.transAxes,
        )
    ax.set_title(f"{task_name.title()} Primary Metric Comparison")
    ax.set_ylabel(primary_metric)
    ax.set_xlabel("Run")
    ax.margins(x=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def _is_obsolete_run(*, run_dir: Path, payload: dict[str, object]) -> bool:
    run_id = str(payload.get("run_id", "")).strip().lower()
    normalized_parts = [part.lower() for part in run_dir.parts]
    if run_id in OBSOLETE_RUN_MARKERS:
        return True
    return any(
        marker in part
        for part in normalized_parts
        for marker in OBSOLETE_EXPERIMENT_MARKERS
    )


def _load_history(history_path: Path) -> list[dict[str, float | int]]:
    return [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
