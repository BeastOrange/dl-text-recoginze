from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dltr.models.detection.trainer import _build_dbnet_tiny, _import_torch, _select_device
from dltr.torch_checkpoint import load_torch_checkpoint


@dataclass(frozen=True)
class DetectionPrediction:
    polygon: list[int]
    score: float


def predict_text_regions(
    *,
    image_path: Path,
    checkpoint_path: Path,
    threshold: float = 0.5,
    min_area: float = 32.0,
) -> list[DetectionPrediction]:
    torch = _import_torch()
    checkpoint = load_torch_checkpoint(torch, checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    image_height = int(config.get("image_height", 256))
    image_width = int(config.get("image_width", 256))
    device = _select_device(torch, str(config.get("device", "auto")))

    model = _build_dbnet_tiny(torch.nn)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    original_height, original_width = image.shape[:2]
    resized = cv2.resize(image, (image_width, image_height))
    tensor = (
        torch.tensor(
            np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1)),
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()

    binary = (probs >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    predictions: list[DetectionPrediction] = []
    scale_x = original_width / max(image_width, 1)
    scale_y = original_height / max(image_height, 1)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        polygon: list[int] = []
        for x, y in box:
            polygon.extend([int(round(x * scale_x)), int(round(y * scale_y))])
        mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
        score = float(probs[mask == 1].mean()) if np.any(mask == 1) else 0.0
        predictions.append(DetectionPrediction(polygon=polygon, score=score))

    predictions.sort(key=lambda item: (min(item.polygon[1::2]), min(item.polygon[0::2])))
    return predictions
