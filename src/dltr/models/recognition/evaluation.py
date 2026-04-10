from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class RecognitionMetrics:
    samples: int
    word_accuracy: float
    cer: float
    ned: float
    mean_edit_distance: float
    latency_ms: float | None = None

    def validate(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be > 0")
        if not 0.0 <= self.word_accuracy <= 1.0:
            raise ValueError("word_accuracy must be in [0, 1]")
        if self.cer < 0:
            raise ValueError("cer must be >= 0")
        if not 0.0 <= self.ned <= 1.0:
            raise ValueError("ned must be in [0, 1]")
        if self.mean_edit_distance < 0:
            raise ValueError("mean_edit_distance must be >= 0")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0 when provided")


def generate_recognition_evaluation_report(
    run_name: str,
    model_name: str,
    metrics: RecognitionMetrics,
    output_dir: str | Path,
    notes: str | None = None,
) -> Path:
    if not run_name.strip():
        raise ValueError("run_name must be non-empty")
    if not model_name.strip():
        raise ValueError("model_name must be non-empty")
    metrics.validate()

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    report_path = target_dir / f"{run_name}_recognition_eval.md"
    timestamp = datetime.now(UTC).isoformat()
    lines = [
        f"# Recognition Evaluation: {run_name}",
        "",
        f"- Generated at (UTC): `{timestamp}`",
        f"- Model: `{model_name}`",
        f"- Samples: `{metrics.samples}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Word Accuracy | {metrics.word_accuracy:.4f} |",
        f"| CER | {metrics.cer:.4f} |",
        f"| NED | {metrics.ned:.4f} |",
        f"| Mean Edit Distance | {metrics.mean_edit_distance:.4f} |",
    ]
    if metrics.latency_ms is not None:
        lines.append(f"| Latency (ms / sample) | {metrics.latency_ms:.4f} |")

    if notes:
        lines.extend(["", "## Notes", "", notes.strip()])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_recognition_evaluation_bundle(
    *,
    run_name: str,
    model_name: str,
    metrics: RecognitionMetrics,
    output_dir: str | Path,
    notes: str | None = None,
    benchmark_name: str | None = None,
    benchmark_category: str | None = None,
) -> dict[str, Path]:
    report_path = generate_recognition_evaluation_report(
        run_name=run_name,
        model_name=model_name,
        metrics=metrics,
        output_dir=output_dir,
        notes=notes,
    )
    target_dir = Path(output_dir)
    json_path = target_dir / f"{run_name}_recognition_eval.json"
    json_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "model_name": model_name,
                "benchmark_name": (benchmark_name or "").strip(),
                "benchmark_category": (benchmark_category or "").strip().lower(),
                "notes": notes.strip() if notes else "",
                "generated_at": datetime.now(UTC).isoformat(),
                "metrics": {
                    "samples": metrics.samples,
                    "word_accuracy": metrics.word_accuracy,
                    "cer": metrics.cer,
                    "ned": metrics.ned,
                    "mean_edit_distance": metrics.mean_edit_distance,
                    "latency_ms": metrics.latency_ms,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"markdown": report_path, "json": json_path}
