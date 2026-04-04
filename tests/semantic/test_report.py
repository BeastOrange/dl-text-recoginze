from pathlib import Path

import pytest

from dltr.semantic.report import SemanticPrediction, generate_semantic_report
from dltr.semantic.slots import extract_semantic_slots


def test_generate_semantic_report_writes_distribution(tmp_path: Path) -> None:
    predictions = [
        SemanticPrediction(
            source_id="img_001",
            text="前方施工 注意安全",
            semantic_class="traffic_or_warning",
            confidence=0.93,
            slots=extract_semantic_slots("前方施工 注意安全"),
        ),
        SemanticPrediction(
            source_id="img_002",
            text="新品促销 买一送一",
            semantic_class="advertisement",
            confidence=0.89,
            slots=extract_semantic_slots("新品促销 买一送一"),
        ),
    ]
    report_path = generate_semantic_report("semantic_demo", predictions, tmp_path)
    content = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "Class Distribution" in content
    assert "advertisement" in content


def test_semantic_prediction_rejects_invalid_class() -> None:
    prediction = SemanticPrediction(
        source_id="img_003",
        text="测试文本",
        semantic_class="unknown_label",
        confidence=0.7,
        slots=extract_semantic_slots("测试文本"),
    )
    with pytest.raises(ValueError, match="Unsupported semantic class"):
        prediction.validate()
