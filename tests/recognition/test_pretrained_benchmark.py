import json
from pathlib import Path

import cv2
import numpy as np

from dltr.models.recognition.pretrained_benchmark import (
    PretrainedRecognitionBackend,
    evaluate_recognition_manifest_with_backend,
)


class _FakeBackend(PretrainedRecognitionBackend):
    def __init__(self, predictions: list[str]) -> None:
        self._predictions = predictions

    def recognize_images(self, image_paths: list[Path]) -> list[tuple[str, float]]:
        return [(text, 0.9) for text in self._predictions[: len(image_paths)]]


def test_evaluate_recognition_manifest_with_backend_computes_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    cv2.imwrite(str(image_a), np.full((32, 96), 255, dtype=np.uint8))
    cv2.imwrite(str(image_b), np.full((32, 96), 255, dtype=np.uint8))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "iiit5k_test",
                        "image_path": str(image_a),
                        "label_path": "",
                        "text": "OPEN",
                    }
                ),
                json.dumps(
                    {
                        "dataset": "iiit5k_test",
                        "image_path": str(image_b),
                        "label_path": "",
                        "text": "AI",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "dltr.models.recognition.pretrained_benchmark._build_backend_session",
        lambda **_: _FakeBackend(["OPEN", "A1"]),
    )

    result = evaluate_recognition_manifest_with_backend(
        manifest_path=manifest,
        backend_name="easyocr",
        device="cpu",
    )

    assert result.metrics.samples == 2
    assert result.metrics.word_accuracy == 0.5
    assert result.metrics.cer > 0.0
    assert len(result.predictions) == 2


def test_evaluate_recognition_manifest_with_backend_supports_upper_normalization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_a = tmp_path / "a.png"
    cv2.imwrite(str(image_a), np.full((32, 96), 255, dtype=np.uint8))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "iiit5k_test",
                "image_path": str(image_a),
                "label_path": "",
                "text": "OPENAI",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "dltr.models.recognition.pretrained_benchmark._build_backend_session",
        lambda **_: _FakeBackend(["openai"]),
    )

    result = evaluate_recognition_manifest_with_backend(
        manifest_path=manifest,
        backend_name="easyocr",
        device="cpu",
        normalize_mode="upper",
    )

    assert result.metrics.word_accuracy == 1.0
