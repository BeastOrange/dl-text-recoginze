from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dltr.semantic.classes import SEMANTIC_CLASSES


@dataclass(frozen=True)
class SemanticSample:
    source_id: str
    text: str
    semantic_class: str


def load_semantic_samples(manifest_path: str | Path) -> list[SemanticSample]:
    path = Path(manifest_path)
    samples: list[SemanticSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        text = str(payload.get("text", "")).strip()
        semantic_class = str(payload.get("semantic_class", "")).strip()
        if not text or semantic_class not in SEMANTIC_CLASSES:
            continue
        samples.append(
            SemanticSample(
                source_id=str(payload.get("source_id", "")).strip() or "unknown",
                text=text,
                semantic_class=semantic_class,
            )
        )
    return samples
