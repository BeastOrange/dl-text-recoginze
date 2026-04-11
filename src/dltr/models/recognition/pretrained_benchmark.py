from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from dltr.models.recognition.dataset import load_recognition_samples
from dltr.models.recognition.evaluation import RecognitionMetrics
from dltr.models.recognition.metrics import compute_recognition_scores


class PretrainedRecognitionBackend:
    def recognize_images(self, image_paths: list[Path]) -> list[tuple[str, float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class ManifestEvaluationResult:
    metrics: RecognitionMetrics
    predictions: list[dict[str, object]]


def evaluate_recognition_manifest_with_backend(
    *,
    manifest_path: str | Path,
    backend_name: str,
    device: str = "auto",
    max_samples: int | None = None,
    normalize_mode: str = "none",
) -> ManifestEvaluationResult:
    samples = load_recognition_samples(manifest_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("No recognition samples available for benchmark evaluation")

    backend = _build_backend_session(backend_name=backend_name, device=device)
    image_paths = [sample.image_path for sample in samples]
    started_at = time.perf_counter()
    outputs = backend.recognize_images(image_paths)
    latency_ms = ((time.perf_counter() - started_at) / len(samples)) * 1000.0
    predictions = [text for text, _ in outputs]
    targets = [sample.text for sample in samples]
    normalized_predictions = [_normalize_text(text, normalize_mode) for text in predictions]
    normalized_targets = [_normalize_text(text, normalize_mode) for text in targets]
    summary = compute_recognition_scores(predictions, targets)
    if normalize_mode != "none":
        summary = compute_recognition_scores(normalized_predictions, normalized_targets)
    metrics = RecognitionMetrics(
        samples=summary.samples,
        word_accuracy=summary.word_accuracy,
        cer=summary.cer,
        ned=summary.ned,
        mean_edit_distance=summary.mean_edit_distance,
        latency_ms=latency_ms,
    )
    records = [
        {
            "dataset": sample.dataset,
            "image_path": str(sample.image_path),
            "target": sample.text,
            "normalized_target": _normalize_text(sample.text, normalize_mode),
            "prediction": prediction,
            "normalized_prediction": _normalize_text(prediction, normalize_mode),
            "confidence": confidence,
        }
        for sample, (prediction, confidence) in zip(samples, outputs, strict=True)
    ]
    return ManifestEvaluationResult(metrics=metrics, predictions=records)


def _build_backend_session(*, backend_name: str, device: str) -> PretrainedRecognitionBackend:
    normalized = backend_name.strip().lower()
    if normalized == "easyocr":
        return _EasyOCRBackend(device=device)
    if normalized == "paddleocr":
        return _PaddleOCRBackend(device=device)
    raise ValueError(f"Unsupported pretrained recognition backend: {backend_name}")


class _EasyOCRBackend(PretrainedRecognitionBackend):
    def __init__(self, *, device: str) -> None:
        try:
            import easyocr
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "EasyOCR is not installed. Use `uv run --with easyocr python ...` "
                "or install it in the runtime environment."
            ) from exc
        gpu = _resolve_easyocr_gpu(torch=torch, device=device)
        self._reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            detector=False,
            recognizer=True,
            verbose=False,
        )

    def recognize_images(self, image_paths: list[Path]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                results.append(("", 0.0))
                continue
            height, width = image.shape[:2]
            detections = self._reader.recognize(
                image,
                horizontal_list=[[0, width, 0, height]],
                free_list=[],
                detail=1,
                paragraph=False,
            )
            if not detections:
                results.append(("", 0.0))
                continue
            _, text, confidence = detections[0]
            results.append((str(text).strip(), float(confidence)))
        return results


class _PaddleOCRBackend(PretrainedRecognitionBackend):
    def __init__(self, *, device: str) -> None:
        try:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
            from paddleocr import TextRecognition
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install `paddlepaddle` and `paddleocr` in the "
                "runtime environment before using this backend."
            ) from exc

        kwargs: dict[str, object] = {"model_name": "PP-OCRv5_server_rec"}
        if device == "cpu":
            kwargs["device"] = "cpu"
        self._model = TextRecognition(**kwargs)

    def recognize_images(self, image_paths: list[Path]) -> list[tuple[str, float]]:
        results = list(self._model.predict([str(path) for path in image_paths], batch_size=16))
        return [
            (
                str(item.get("rec_text", "")).strip(),
                float(item.get("rec_score", 0.0)),
            )
            for item in results
        ]


def _resolve_easyocr_gpu(*, torch: Any, device: str) -> bool:
    if device == "cpu":
        return False
    if device == "cuda":
        return True
    return bool(torch.cuda.is_available())


def _normalize_text(text: str, mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "none":
        return text.strip()
    if normalized == "upper":
        return text.strip().upper()
    if normalized == "alnum_upper":
        return re.sub(r"[^0-9A-Z]", "", text.strip().upper())
    raise ValueError(f"Unsupported normalize mode: {mode}")
