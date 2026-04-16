from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from dltr.models.recognition.charset import CharacterVocabulary
from dltr.models.recognition.dataset import RecognitionSample


@dataclass(frozen=True)
class SplitDiagnostics:
    split_name: str
    sample_count: int
    dataset_counts: dict[str, int]
    empty_label_count: int
    total_characters: int
    in_vocab_characters: int
    oov_characters: int
    oov_char_ratio: float
    oov_top_characters: list[tuple[str, int]]


@dataclass(frozen=True)
class RecognitionTrainingDiagnostics:
    train: SplitDiagnostics
    validation: SplitDiagnostics
    charset_size: int
    charset_path: str


def build_split_diagnostics(
    *,
    split_name: str,
    samples: list[RecognitionSample],
    vocabulary: CharacterVocabulary,
    top_k: int = 20,
) -> SplitDiagnostics:
    dataset_counter: Counter[str] = Counter()
    oov_counter: Counter[str] = Counter()
    vocabulary_set = set(vocabulary.characters)
    empty_label_count = 0
    total_characters = 0
    in_vocab_characters = 0
    oov_characters = 0

    for sample in samples:
        dataset_counter[sample.dataset] += 1
        if not sample.text:
            empty_label_count += 1
            continue
        for character in sample.text:
            total_characters += 1
            if character not in vocabulary_set:
                oov_characters += 1
                oov_counter[character] += 1
            else:
                in_vocab_characters += 1

    ratio = oov_characters / total_characters if total_characters else 0.0
    return SplitDiagnostics(
        split_name=split_name,
        sample_count=len(samples),
        dataset_counts=dict(dataset_counter),
        empty_label_count=empty_label_count,
        total_characters=total_characters,
        in_vocab_characters=in_vocab_characters,
        oov_characters=oov_characters,
        oov_char_ratio=ratio,
        oov_top_characters=oov_counter.most_common(top_k),
    )


def build_training_diagnostics(
    *,
    train_samples: list[RecognitionSample],
    val_samples: list[RecognitionSample],
    vocabulary: CharacterVocabulary,
    charset_path: Path,
    top_k: int = 20,
) -> RecognitionTrainingDiagnostics:
    return RecognitionTrainingDiagnostics(
        train=build_split_diagnostics(
            split_name="train",
            samples=train_samples,
            vocabulary=vocabulary,
            top_k=top_k,
        ),
        validation=build_split_diagnostics(
            split_name="validation",
            samples=val_samples,
            vocabulary=vocabulary,
            top_k=top_k,
        ),
        charset_size=vocabulary.size,
        charset_path=str(charset_path),
    )


def write_training_diagnostics(
    diagnostics: RecognitionTrainingDiagnostics,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "training_diagnostics.json"
    markdown_path = output_dir / "training_diagnostics.md"
    json_path.write_text(
        json.dumps(asdict(diagnostics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(_build_markdown(diagnostics), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _build_markdown(diagnostics: RecognitionTrainingDiagnostics) -> str:
    lines = [
        "# Recognition Training Diagnostics",
        "",
        f"- Charset Size: `{diagnostics.charset_size}`",
        f"- Charset Path: `{diagnostics.charset_path}`",
        "",
    ]
    for split in (diagnostics.train, diagnostics.validation):
        lines.extend(
            [
                f"## {split.split_name.title()}",
                "",
                f"- Samples: `{split.sample_count}`",
                f"- Empty Labels: `{split.empty_label_count}`",
                f"- Total Characters: `{split.total_characters}`",
                f"- In-Vocab Characters: `{split.in_vocab_characters}`",
                f"- OOV Characters: `{split.oov_characters}`",
                f"- OOV Ratio: `{split.oov_char_ratio:.6f}`",
                "",
                "### Dataset Counts",
                "",
            ]
        )
        if split.dataset_counts:
            lines.extend(
                f"- `{name}`: `{count}`" for name, count in sorted(split.dataset_counts.items())
            )
        else:
            lines.append("- `<empty>`: `0`")
        lines.extend(["", "### OOV Top Characters", ""])
        if split.oov_top_characters:
            lines.extend(
                f"- `{char}`: `{count}`"
                for char, count in split.oov_top_characters
            )
        else:
            lines.append("- `<none>`: `0`")
        lines.append("")
    return "\n".join(lines)
