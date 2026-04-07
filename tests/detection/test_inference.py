from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dltr.models.detection.inference import DetectionPredictorSession

torch = pytest.importorskip("torch")


def test_detection_predictor_session_falls_back_to_legacy_architecture_when_missing_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "legacy.pt"
    checkpoint_path.write_bytes(b"pt")

    captured: dict[str, object] = {}

    def fake_load_checkpoint(_torch, _path, *, map_location="cpu"):  # noqa: ANN001
        return {
            "config": {"image_height": 32, "image_width": 32, "device": "cpu"},
            "model_state_dict": {"dummy": 1},
        }

    class _Model:
        def load_state_dict(self, state_dict):  # noqa: ANN001
            captured["state_dict"] = state_dict

        def to(self, device):  # noqa: ANN001
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = True
            return self

    def fake_build_detection_model(_nn, *, architecture: str):  # noqa: ANN001
        captured["architecture"] = architecture
        return _Model()

    monkeypatch.setattr(
        "dltr.models.detection.inference.load_torch_checkpoint",
        fake_load_checkpoint,
    )
    monkeypatch.setattr(
        "dltr.models.detection.inference._build_detection_model",
        fake_build_detection_model,
    )

    session = DetectionPredictorSession.from_checkpoint(checkpoint_path)

    assert session.image_height == 32
    assert session.image_width == 32
    assert captured["architecture"] == "dbnet_tiny"


def test_detection_predictor_session_converts_bgr_image_before_normalization(
    monkeypatch,
) -> None:
    session = DetectionPredictorSession(
        torch=torch,
        model=_StubDetectorModel(torch),
        device="cpu",
        image_height=4,
        image_width=4,
        checkpoint_path=Path("dummy.pt"),
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0] = 255  # BGR blue-only input

    captured: dict[str, np.ndarray] = {}

    def fake_prepare_detection_image(image_array, *, target_height, target_width):  # noqa: ANN001
        captured["image"] = image_array.copy()
        return np.zeros((target_height, target_width, 3), dtype=np.float32)

    monkeypatch.setattr(
        "dltr.models.detection.inference._prepare_detection_image",
        fake_prepare_detection_image,
    )

    session.predict_image(image, threshold=0.5, min_area=32.0)

    assert np.all(captured["image"][..., 2] == 255)
    assert np.all(captured["image"][..., 0] == 0)


class _StubDetectorModel:
    def __init__(self, torch_module):  # noqa: ANN001
        self._torch = torch_module

    def __call__(self, tensor):  # noqa: ANN001
        return self._torch.zeros((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]))

