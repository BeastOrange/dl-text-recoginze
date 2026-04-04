from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecognitionScoreSummary:
    samples: int
    word_accuracy: float
    cer: float
    ned: float
    mean_edit_distance: float


def compute_recognition_scores(
    predictions: list[str],
    targets: list[str],
) -> RecognitionScoreSummary:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not predictions:
        raise ValueError("predictions must not be empty")

    exact_matches = 0
    total_distance = 0
    total_characters = 0
    normalized_distances: list[float] = []

    for prediction, target in zip(predictions, targets, strict=True):
        distance = _edit_distance(prediction, target)
        total_distance += distance
        total_characters += max(1, len(target))
        if prediction == target:
            exact_matches += 1
        normalized_distances.append(distance / max(len(prediction), len(target), 1))

    samples = len(predictions)
    return RecognitionScoreSummary(
        samples=samples,
        word_accuracy=exact_matches / samples,
        cer=total_distance / total_characters,
        ned=sum(normalized_distances) / samples,
        mean_edit_distance=total_distance / samples,
    )


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            insert_cost = current[column_index - 1] + 1
            delete_cost = previous[column_index] + 1
            replace_cost = previous[column_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]
