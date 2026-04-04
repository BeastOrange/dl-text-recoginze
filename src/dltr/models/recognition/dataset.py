from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecognitionSample:
    dataset: str
    image_path: Path
    label_path: Path
    text: str


def load_recognition_samples(manifest_path: str | Path) -> list[RecognitionSample]:
    path = Path(manifest_path)
    samples: list[RecognitionSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        text = str(payload.get("text", "")).strip()
        image_path = Path(str(payload.get("image_path", "")))
        if not text or not image_path.exists():
            continue
        samples.append(
            RecognitionSample(
                dataset=str(payload.get("dataset", "")).strip(),
                image_path=image_path,
                label_path=Path(str(payload.get("label_path", ""))),
                text=text,
            )
        )
    return samples
