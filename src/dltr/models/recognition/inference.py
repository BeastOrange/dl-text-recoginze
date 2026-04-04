from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dltr.models.recognition.charset import CharacterVocabulary
from dltr.models.recognition.trainer import _build_crnn_model, _import_torch, _select_device


@dataclass(frozen=True)
class RecognitionPrediction:
    text: str
    confidence: float


def recognize_crop(
    *,
    image_path: Path,
    checkpoint_path: Path,
) -> RecognitionPrediction:
    torch = _import_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    image_height = int(config.get("image_height", 48))
    image_width = int(config.get("image_width", 320))
    charset_file = Path(str(config.get("charset_file", "")))
    if not charset_file.is_absolute():
        charset_file = Path.cwd() / charset_file
    vocabulary = CharacterVocabulary.from_file(charset_file)
    device = _select_device(torch, str(config.get("device", "auto")))

    model = _build_crnn_model(torch.nn, vocabulary_size=vocabulary.size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read crop image: {image_path}")
    image = cv2.resize(image, (image_width, image_height))
    tensor = (
        torch.tensor(image.astype(np.float32) / 255.0, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        log_probs = model(tensor)
        probs = torch.exp(log_probs)
        greedy = probs.argmax(dim=2).permute(1, 0)[0]
        max_probs = probs.max(dim=2).values.permute(1, 0)[0]

    return RecognitionPrediction(
        text=vocabulary.decode_greedy(greedy.tolist()),
        confidence=float(max_probs.mean().item()) if max_probs.numel() else 0.0,
    )
