from dltr.semantic.slots import extract_semantic_slots


def test_extract_semantic_slots_parses_chinese_text() -> None:
    text = "地址：中山路88号，电话13800138000，营业时间09:00-18:00，特价¥29.9元，严禁烟火。"
    slots = extract_semantic_slots(text)

    assert "13800138000" in slots.phone
    assert any(item.startswith("¥29.9") for item in slots.price)
    assert "09:00" in slots.time
    assert "严禁" in slots.warning_terms
    assert "路" in slots.location_hint
    assert "中山路" in slots.keywords
