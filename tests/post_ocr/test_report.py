from pathlib import Path

import pytest

from dltr.post_ocr.report import PostOCRPrediction, generate_post_ocr_report
from dltr.post_ocr.slots import extract_post_ocr_slots


def test_generate_post_ocr_report_writes_distribution(tmp_path: Path) -> None:
    predictions = [
        PostOCRPrediction(
            source_id="img_001",
            text="前方施工 注意安全",
            analysis_label="traffic_or_warning",
            confidence=0.93,
            slots=extract_post_ocr_slots("前方施工 注意安全"),
        ),
        PostOCRPrediction(
            source_id="img_002",
            text="新品促销 买一送一",
            analysis_label="advertisement",
            confidence=0.89,
            slots=extract_post_ocr_slots("新品促销 买一送一"),
        ),
    ]
    report_path = generate_post_ocr_report("demo", predictions, tmp_path)
    content = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "Analysis Label Distribution" in content
    assert "advertisement" in content


def test_post_ocr_prediction_rejects_invalid_label() -> None:
    prediction = PostOCRPrediction(
        source_id="img_003",
        text="测试文本",
        analysis_label="unknown_label",
        confidence=0.7,
        slots=extract_post_ocr_slots("测试文本"),
    )
    with pytest.raises(ValueError, match="Unsupported analysis label"):
        prediction.validate()
