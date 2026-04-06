from __future__ import annotations

from pathlib import Path
from typing import Any


def load_torch_checkpoint(
    torch: Any,
    checkpoint_path: Path,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(
            checkpoint_path,
            map_location=map_location,
        )
