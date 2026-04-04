import numpy as np

from dltr.models.detection.metrics import compute_detection_scores


def test_compute_detection_scores_reports_reasonable_values() -> None:
    prediction = np.array([[[[0.9, 0.8], [0.1, 0.2]]]], dtype=np.float32)
    target = np.array([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=np.float32)

    scores = compute_detection_scores(prediction, target, threshold=0.5)

    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["hmean"] == 1.0
