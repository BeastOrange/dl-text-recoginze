from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dltr.models.detection.trainer import _build_dbnet_tiny, _import_torch, _select_device
from dltr.torch_checkpoint import load_torch_checkpoint


@dataclass(frozen=True)
class DetectionPrediction:
    polygon: list[int]
    score: float


@dataclass(frozen=True)
class DetectionPredictorSession:
    torch: Any
    model: Any
    device: str
    image_height: int
    image_width: int
    checkpoint_path: Path

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> DetectionPredictorSession:
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
        return cls(
            torch=torch,
            model=model,
            device=device,
            image_height=image_height,
            image_width=image_width,
            checkpoint_path=checkpoint_path,
        )

    def predict_path(
        self,
        image_path: Path,
        *,
        threshold: float = 0.5,
        min_area: float = 32.0,
    ) -> list[DetectionPrediction]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.predict_image(
            image,
            threshold=threshold,
            min_area=min_area,
        )

    def predict_image(
        self,
        image: np.ndarray,
        *,
        threshold: float = 0.5,
        min_area: float = 32.0,
    ) -> list[DetectionPrediction]:
        original_height, original_width = image.shape[:2]
        resized = cv2.resize(image, (self.image_width, self.image_height))
        tensor = (
            self.torch.tensor(
                np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1)),
                dtype=self.torch.float32,
            )
            .unsqueeze(0)
            .to(self.device)
        )

        with self.torch.no_grad():
            probs = self.torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()

        return _decode_detection_map(
            probs=probs,
            threshold=threshold,
            min_area=min_area,
            original_width=original_width,
            original_height=original_height,
            model_width=self.image_width,
            model_height=self.image_height,
        )


def predict_text_regions(
    *,
    image_path: Path,
    checkpoint_path: Path,
    threshold: float = 0.5,
    min_area: float = 32.0,
) -> list[DetectionPrediction]:
    session = DetectionPredictorSession.from_checkpoint(checkpoint_path)
    return session.predict_path(
        image_path,
        threshold=threshold,
        min_area=min_area,
    )


def _decode_detection_map(
    *,
    probs: np.ndarray,
    threshold: float,
    min_area: float,
    original_width: int,
    original_height: int,
    model_width: int,
    model_height: int,
) -> list[DetectionPrediction]:
    binary = (probs >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    predictions: list[DetectionPrediction] = []
    scale_x = original_width / max(model_width, 1)
    scale_y = original_height / max(model_height, 1)

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
