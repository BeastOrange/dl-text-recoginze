from __future__ import annotations

import json
from pathlib import Path

from dltr.semantic.classification import classify_scene_text


def build_semantic_manifests_from_recognition(
    *,
    recognition_split_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        source_path = recognition_split_dir / f"{split}.jsonl"
        target_path = output_dir / f"{split}.jsonl"
        rows: list[str] = []
        for raw_line in source_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            text = str(payload.get("text", "")).strip()
            if not text:
                continue
            semantic = classify_scene_text(text)
            rows.append(
                json.dumps(
                    {
                        "source_id": f"{split}-{len(rows)}",
                        "text": text,
                        "semantic_class": semantic.semantic_class,
                    },
                    ensure_ascii=False,
                )
            )
        target_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        outputs[split] = target_path
    return outputs
