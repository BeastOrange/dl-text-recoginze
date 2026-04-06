from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dltr.post_ocr.classification import validate_analysis_label
from dltr.post_ocr.slots import PostOCRSlots


@dataclass(frozen=True)
class PostOCRPrediction:
    source_id: str
    text: str
    analysis_label: str
    confidence: float
    slots: PostOCRSlots

    def validate(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        validate_analysis_label(self.analysis_label)


def generate_post_ocr_report(
    run_name: str,
    predictions: list[PostOCRPrediction],
    output_dir: str | Path,
) -> Path:
    if not run_name.strip():
        raise ValueError("run_name must be non-empty")
    if not predictions:
        raise ValueError("predictions must not be empty")
    for prediction in predictions:
        prediction.validate()

    counts = Counter(pred.analysis_label for pred in predictions)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / f"{run_name}_analysis_report.md"
    timestamp = datetime.now(UTC).isoformat()

    lines = [
        f"# Post-OCR Analysis Report: {run_name}",
        "",
        f"- Generated at (UTC): `{timestamp}`",
        f"- Samples: `{len(predictions)}`",
        "",
        "## Analysis Label Distribution",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(counts.items(), key=lambda item: item[0]):
        lines.append(f"| {label} | {count} |")

    lines.extend(
        [
            "",
            "## Prediction Preview",
            "",
            "| Source | Label | Confidence | Keywords | Phone | Price | Time |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for sample in predictions[:10]:
        keywords = ", ".join(sample.slots.keywords[:4]) or "-"
        phone = ", ".join(sample.slots.phone[:2]) or "-"
        price = ", ".join(sample.slots.price[:2]) or "-"
        time = ", ".join(sample.slots.time[:2]) or "-"
        lines.append(
            f"| {sample.source_id} | {sample.analysis_label} | {sample.confidence:.4f} | "
            f"{keywords} | {phone} | {price} | {time} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
