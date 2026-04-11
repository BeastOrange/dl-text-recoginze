from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from dltr.visualization.plot_style import (
    bar_colors,
    resolve_label_rotation,
    resolve_summary_fig_width,
    resolve_upper_bound,
    style_axis,
)


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark: str
    category: str
    word_accuracy: float
    samples: int
    cer: float = 0.0
    ned: float = 0.0
    mean_edit_distance: float = 0.0
    run_name: str = ""
    model_name: str = ""
    source_json: str = ""


def build_english_benchmark_summary(
    *,
    output_dir: Path,
    benchmark_json_paths: list[Path] | None = None,
    records: list[BenchmarkRecord] | None = None,
    report_name: str = "english_benchmark_summary",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_records = _resolve_records(
        benchmark_json_paths=benchmark_json_paths or [],
        records=records or [],
    )
    main_records = [item for item in resolved_records if item.category == "main"]
    hard_records = [item for item in resolved_records if item.category == "hard"]
    payload = {
        "summary": {
            "main_average_word_accuracy": _average_metric(main_records, "word_accuracy"),
            "hard_average_word_accuracy": _average_metric(hard_records, "word_accuracy"),
            "main_benchmark_count": len(main_records),
            "hard_benchmark_count": len(hard_records),
        },
        "benchmarks": [_record_to_dict(item) for item in resolved_records],
    }
    json_path = output_dir / f"{report_name}.json"
    markdown_path = output_dir / f"{report_name}.md"
    png_path = output_dir / f"{report_name}.png"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_build_markdown(payload, png_name=png_path.name), encoding="utf-8")
    _render_benchmark_plot(records=resolved_records, png_path=png_path)
    return {"json": json_path, "markdown": markdown_path, "png": png_path}


def _resolve_records(
    *,
    benchmark_json_paths: list[Path],
    records: list[BenchmarkRecord],
) -> list[BenchmarkRecord]:
    if records:
        return records
    return [_load_benchmark_record(path) for path in benchmark_json_paths]


def _load_benchmark_record(path: Path) -> BenchmarkRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    return BenchmarkRecord(
        benchmark=str(payload.get("benchmark_name", "")).strip(),
        category=str(payload.get("benchmark_category", "")).strip().lower(),
        word_accuracy=float(metrics.get("word_accuracy", 0.0)),
        samples=int(metrics.get("samples", 0)),
        cer=float(metrics.get("cer", 0.0)),
        ned=float(metrics.get("ned", 0.0)),
        mean_edit_distance=float(metrics.get("mean_edit_distance", 0.0)),
        run_name=str(payload.get("run_name", "")).strip(),
        model_name=str(payload.get("model_name", "")).strip(),
        source_json=str(path),
    )


def _average_metric(records: list[BenchmarkRecord], metric_name: str) -> float | None:
    if not records:
        return None
    return sum(float(getattr(item, metric_name)) for item in records) / len(records)


def _record_to_dict(record: BenchmarkRecord) -> dict[str, object]:
    return {
        "benchmark": record.benchmark,
        "category": record.category,
        "word_accuracy": record.word_accuracy,
        "samples": record.samples,
        "cer": record.cer,
        "ned": record.ned,
        "mean_edit_distance": record.mean_edit_distance,
        "run_name": record.run_name,
        "model_name": record.model_name,
        "source_json": record.source_json,
    }


def _build_markdown(payload: dict[str, object], *, png_name: str) -> str:
    summary = payload["summary"]
    records = payload["benchmarks"]
    lines = [
        "# English Benchmark Summary",
        "",
        f"- Main-English-Accuracy: `{_format_metric(summary['main_average_word_accuracy'])}`",
        f"- Hard-English-Accuracy: `{_format_metric(summary['hard_average_word_accuracy'])}`",
        f"- Plot: `{png_name}`",
        "",
        "| Benchmark | Category | Word Accuracy | Samples | CER | NED |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            f"| {item['benchmark']} | {item['category']} | "
            f"{float(item['word_accuracy']):.6f} | {int(item['samples'])} | "
            f"{float(item['cer']):.6f} | {float(item['ned']):.6f} |"
        )
    return "\n".join(lines) + "\n"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def _render_benchmark_plot(*, records: list[BenchmarkRecord], png_path: Path) -> None:
    fig_width = resolve_summary_fig_width([item.benchmark for item in records])
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    style_axis(ax)
    if records:
        labels = [item.benchmark for item in records]
        values = [item.word_accuracy for item in records]
        x_positions = list(range(len(records)))
        rotation = resolve_label_rotation(labels)
        bars = ax.bar(
            x_positions,
            values,
            color=bar_colors(len(records), cmap_name="Greens"),
            width=0.62,
            edgecolor="#2E3A46",
            linewidth=0.6,
        )
        upper_bound = resolve_upper_bound(max(values))
        ax.set_ylim(0.0, upper_bound)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
        ax.set_ylabel("Word Accuracy")
        ax.set_title("English Benchmark Accuracy")
        for bar, value, record in zip(bars, values, records, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper_bound * 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#2E3A46",
            )
            if record.category == "hard":
                bar.set_hatch("//")
    else:
        ax.text(0.5, 0.5, "No benchmark records", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
