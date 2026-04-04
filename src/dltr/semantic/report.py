from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .classes import validate_semantic_class
from .slots import SemanticSlots


@dataclass(frozen=True)
class SemanticPrediction:
    source_id: str
    text: str
    semantic_class: str
    confidence: float
    slots: SemanticSlots

    def validate(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        validate_semantic_class(self.semantic_class)


def generate_semantic_report(
    run_name: str,
    predictions: list[SemanticPrediction],
    output_dir: str | Path,
) -> Path:
    if not run_name.strip():
        raise ValueError("run_name must be non-empty")
    if not predictions:
        raise ValueError("predictions must not be empty")
    for prediction in predictions:
        prediction.validate()

    counts = Counter(pred.semantic_class for pred in predictions)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / f"{run_name}_semantic_eval.md"
    timestamp = datetime.now(UTC).isoformat()

    lines = [
        f"# Semantic Evaluation: {run_name}",
        "",
        f"- Generated at (UTC): `{timestamp}`",
        f"- Samples: `{len(predictions)}`",
        "",
        "## Class Distribution",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(counts.items(), key=lambda item: item[0]):
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Prediction Preview",
            "",
            "| Source | Class | Confidence | Keywords | Phone | Price | Time |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for sample in predictions[:10]:
        keywords = ", ".join(sample.slots.keywords[:4]) or "-"
        phone = ", ".join(sample.slots.phone[:2]) or "-"
        price = ", ".join(sample.slots.price[:2]) or "-"
        time = ", ".join(sample.slots.time[:2]) or "-"
        lines.append(
            f"| {sample.source_id} | {sample.semantic_class} | {sample.confidence:.4f} | "
            f"{keywords} | {phone} | {price} | {time} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
