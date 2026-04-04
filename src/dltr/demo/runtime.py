from __future__ import annotations

import tempfile
from pathlib import Path

from dltr.pipeline.checkpoints import discover_latest_run_dir, resolve_best_checkpoint
from dltr.pipeline.end_to_end import EndToEndPipelineArtifacts, run_end_to_end_pipeline


def resolve_demo_checkpoints(
    *,
    project_root: Path,
    detector_run_dir: Path | None = None,
    recognizer_run_dir: Path | None = None,
) -> dict[str, Path]:
    detection_run = detector_run_dir or discover_latest_run_dir(
        project_root / "artifacts" / "detection"
    )
    recognition_run = recognizer_run_dir or discover_latest_run_dir(
        project_root / "artifacts" / "checkpoints" / "recognition"
    )
    return {
        "detector": resolve_best_checkpoint(detection_run),
        "recognizer": resolve_best_checkpoint(recognition_run),
    }


def run_uploaded_inference(
    *,
    image_bytes: bytes,
    project_root: Path,
    detector_checkpoint: Path,
    recognizer_checkpoint: Path,
) -> EndToEndPipelineArtifacts:
    uploads_dir = project_root / "reports" / "demo_assets" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=uploads_dir, suffix=".png", delete=False) as handle:
        handle.write(image_bytes)
        temp_path = Path(handle.name)

    output_dir = project_root / "reports" / "eval" / "interactive"
    return run_end_to_end_pipeline(
        image_path=temp_path,
        output_dir=output_dir,
        detector_checkpoint=detector_checkpoint,
        recognizer_checkpoint=recognizer_checkpoint,
    )
