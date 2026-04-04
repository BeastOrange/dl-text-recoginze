from dltr.semantic.classification import classify_scene_text


def test_classify_scene_text_detects_warning_category() -> None:
    result = classify_scene_text("当心高压 危险")

    assert result.semantic_class == "traffic_or_warning"
    assert result.confidence > 0.0


def test_classify_scene_text_detects_service_info() -> None:
    result = classify_scene_text("营业时间 09:00-21:00 电话 13800138000")

    assert result.semantic_class == "service_info"


def test_classify_scene_text_detects_advertisement() -> None:
    result = classify_scene_text("开业大促 全场五折 特价")

    assert result.semantic_class == "advertisement"
