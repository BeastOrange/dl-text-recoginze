from __future__ import annotations

from pathlib import Path

from dltr.models.detection.trainer import (
    DEFAULT_DETECTION_MODEL_ARCHITECTURE,
    LEGACY_DETECTION_MODEL_ARCHITECTURE,
    _build_detection_model,
    _import_torch,
    _select_device,
)
from dltr.torch_checkpoint import load_torch_checkpoint


def export_detection_model_to_onnx(
    *,
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    torch = _import_torch()
    checkpoint = load_torch_checkpoint(torch, checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    image_height = int(config.get("image_height", 256))
    image_width = int(config.get("image_width", 256))
    raw_architecture = checkpoint.get("model_architecture")
    model_architecture = (
        str(raw_architecture).strip()
        if raw_architecture is not None
        else DEFAULT_DETECTION_MODEL_ARCHITECTURE
    ) or DEFAULT_DETECTION_MODEL_ARCHITECTURE
    if model_architecture not in {
        DEFAULT_DETECTION_MODEL_ARCHITECTURE,
        LEGACY_DETECTION_MODEL_ARCHITECTURE,
    }:
        model_architecture = DEFAULT_DETECTION_MODEL_ARCHITECTURE
    device = _select_device(torch, str(config.get("device", "auto")))
    model = _build_detection_model(torch.nn, architecture=model_architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dummy = torch.randn(1, 3, image_height, image_width, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=17,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX export dependency is missing. "
            "Install `onnx` and `onnxscript` before running export."
        ) from exc
    return output_path
