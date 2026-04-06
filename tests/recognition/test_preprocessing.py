import numpy as np

from dltr.models.recognition.preprocessing import (
    RecognitionPreprocessConfig,
    prepare_recognition_image,
)


def test_prepare_recognition_image_rotates_vertical_text_clockwise() -> None:
    image = np.full((80, 20), 255, dtype=np.uint8)
    image[:20, :] = 0

    processed, meta = prepare_recognition_image(
        image,
        config=RecognitionPreprocessConfig(
            target_height=48,
            target_width=160,
            rotate_vertical_text=True,
            vertical_aspect_threshold=1.2,
        ),
    )

    assert processed.shape == (48, 160)
    assert meta.rotated_clockwise is True
    assert meta.content_width > meta.content_height


def test_prepare_recognition_image_preserves_aspect_ratio_with_right_padding() -> None:
    image = np.zeros((20, 100), dtype=np.uint8)

    processed, meta = prepare_recognition_image(
        image,
        config=RecognitionPreprocessConfig(
            target_height=48,
            target_width=320,
            preserve_aspect_ratio=True,
            rotate_vertical_text=False,
        ),
    )

    assert processed.shape == (48, 320)
    assert meta.content_width < 320
    assert float(processed[:, -1].mean()) == 1.0
    assert float(processed[:, : meta.content_width].mean()) < 0.1
