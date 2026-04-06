from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dltr.models.recognition.charset import CharacterVocabulary
from dltr.models.recognition.preprocessing import (
    RecognitionPreprocessConfig,
    prepare_recognition_image,
)
from dltr.models.recognition.trainer import (
    _build_recognizer_model,
    _import_torch,
    _select_device,
)
from dltr.torch_checkpoint import load_torch_checkpoint


@dataclass(frozen=True)
class RecognitionPrediction:
    text: str
    confidence: float


class RecognitionPredictorSession:
    def __init__(
        self,
        *,
        torch: Any,
        model: Any,
        vocabulary: CharacterVocabulary,
        device: str,
        image_height: int,
        image_width: int,
        preprocess_config: RecognitionPreprocessConfig,
    ) -> None:
        self._torch = torch
        self._model = model
        self._vocabulary = vocabulary
        self._device = device
        self._image_height = image_height
        self._image_width = image_width
        self._preprocess_config = preprocess_config

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> RecognitionPredictorSession:
        torch = _import_torch()
        checkpoint = load_torch_checkpoint(torch, checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model_name = str(config.get("model_name", "crnn")).strip().lower() or "crnn"
        image_height = int(config.get("image_height", 48))
        image_width = int(config.get("image_width", 320))
        charset_file = _resolve_charset_path(
            checkpoint_path=checkpoint_path,
            charset_raw=str(config.get("charset_file", "")),
        )
        vocabulary = CharacterVocabulary.from_file(charset_file)
        device = _select_device(torch, str(config.get("device", "auto")))
        preprocess_raw = config.get("preprocess", {})
        preprocess_config = RecognitionPreprocessConfig(
            target_height=image_height,
            target_width=image_width,
            preserve_aspect_ratio=bool(preprocess_raw.get("preserve_aspect_ratio", True)),
            rotate_vertical_text=bool(preprocess_raw.get("rotate_vertical_text", True)),
            vertical_aspect_threshold=float(
                preprocess_raw.get("vertical_aspect_threshold", 1.2)
            ),
            padding_value=int(preprocess_raw.get("padding_value", 255)),
        )

        model = _build_recognizer_model(torch.nn, model_name, vocabulary_size=vocabulary.size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return cls(
            torch=torch,
            model=model,
            vocabulary=vocabulary,
            device=device,
            image_height=image_height,
            image_width=image_width,
            preprocess_config=preprocess_config,
        )

    def recognize_image(self, image: np.ndarray) -> RecognitionPrediction:
        resized, _ = prepare_recognition_image(image, config=self._preprocess_config)
        tensor = (
            self._torch.tensor(resized, dtype=self._torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self._device)
        )
        return self._decode_tensor(tensor)

    def recognize_images(self, images: list[np.ndarray]) -> list[RecognitionPrediction]:
        if not images:
            return []
        batch = np.stack(
            [
                prepare_recognition_image(image, config=self._preprocess_config)[0]
                for image in images
            ],
            axis=0,
        )
        tensor = (
            self._torch.tensor(batch.astype(np.float32), dtype=self._torch.float32)
            .unsqueeze(1)
            .to(self._device)
        )
        with self._torch.no_grad():
            log_probs = self._model(tensor)
            probs = self._torch.exp(log_probs)
            greedy_batch = probs.argmax(dim=2).permute(1, 0)
            confidence_batch = probs.max(dim=2).values.permute(1, 0)
        predictions: list[RecognitionPrediction] = []
        for greedy, confidence in zip(greedy_batch, confidence_batch, strict=True):
            predictions.append(
                RecognitionPrediction(
                    text=self._vocabulary.decode_greedy(greedy.tolist()),
                    confidence=float(confidence.mean().item()) if confidence.numel() else 0.0,
                )
            )
        return predictions

    def _decode_tensor(self, tensor: Any) -> RecognitionPrediction:
        with self._torch.no_grad():
            log_probs = self._model(tensor)
            probs = self._torch.exp(log_probs)
            greedy = probs.argmax(dim=2).permute(1, 0)[0]
            max_probs = probs.max(dim=2).values.permute(1, 0)[0]
        return RecognitionPrediction(
            text=self._vocabulary.decode_greedy(greedy.tolist()),
            confidence=float(max_probs.mean().item()) if max_probs.numel() else 0.0,
        )


def recognize_crop(
    *,
    image_path: Path,
    checkpoint_path: Path,
) -> RecognitionPrediction:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read crop image: {image_path}")
    session = _get_cached_session(checkpoint_path.resolve())
    return session.recognize_image(image)


@lru_cache(maxsize=4)
def _get_cached_session(checkpoint_path: Path) -> RecognitionPredictorSession:
    return RecognitionPredictorSession.from_checkpoint(checkpoint_path)


def _resolve_charset_path(*, checkpoint_path: Path, charset_raw: str) -> Path:
    charset_file = Path(charset_raw)
    if charset_file.is_absolute():
        return charset_file
    repo_candidate = Path.cwd() / charset_file
    if repo_candidate.exists():
        return repo_candidate
    checkpoint_candidate = checkpoint_path.resolve().parents[3] / charset_file
    return checkpoint_candidate
