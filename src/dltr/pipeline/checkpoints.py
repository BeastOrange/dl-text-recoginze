from __future__ import annotations

import json
from pathlib import Path


def resolve_best_checkpoint(run_dir: Path) -> Path:
    summary_path = run_dir / "training_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        best_checkpoint = payload.get("best_checkpoint_path")
        if best_checkpoint:
            candidate = Path(str(best_checkpoint))
            if candidate.exists():
                return candidate

    for candidate in (
        run_dir / "best.pt",
        run_dir / "checkpoints" / "best.pt",
        run_dir / "last.pt",
        run_dir / "checkpoints" / "last.pt",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint could be resolved from run dir: {run_dir}")


def discover_latest_run_dir(root_dir: Path) -> Path:
    candidates = [
        summary_path.parent
        for summary_path in root_dir.rglob("training_summary.json")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run directories with training_summary.json found in {root_dir}"
        )
    return sorted(candidates, key=lambda item: item.name)[-1]
