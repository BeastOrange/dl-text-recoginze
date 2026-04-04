from dltr.semantic.classes import SEMANTIC_CLASSES, validate_semantic_class


def test_semantic_classes_match_plan_scope() -> None:
    assert "shop_sign" in SEMANTIC_CLASSES
    assert "advertisement" in SEMANTIC_CLASSES
    assert "public_notice" in SEMANTIC_CLASSES
    assert "traffic_or_warning" in SEMANTIC_CLASSES
    assert "service_info" in SEMANTIC_CLASSES
    assert "other" in SEMANTIC_CLASSES


def test_validate_semantic_class_accepts_known_label() -> None:
    assert validate_semantic_class("shop_sign") == "shop_sign"
