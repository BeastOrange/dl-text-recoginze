from __future__ import annotations

import re
from dataclasses import dataclass

WARNING_LEXICON = (
    "危险",
    "高压",
    "严禁",
    "禁止",
    "注意",
    "当心",
    "易燃",
    "有毒",
    "警告",
    "消防",
)

LOCATION_TERMS = (
    "路",
    "街",
    "大道",
    "巷",
    "号",
    "楼",
    "层",
    "室",
    "站",
    "出口",
    "广场",
    "商场",
    "大厦",
    "中心",
    "医院",
    "学校",
    "园区",
    "地铁",
)

STOPWORDS = {
    "本店",
    "本公司",
    "请",
    "的",
    "和",
    "与",
    "在",
    "为",
    "及",
}


@dataclass(frozen=True)
class SemanticSlots:
    phone: list[str]
    price: list[str]
    time: list[str]
    warning_terms: list[str]
    location_hint: list[str]
    keywords: list[str]


def _deduplicate(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def extract_keywords(text: str, top_k: int = 8) -> list[str]:
    chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", text)
    filtered = [token for token in chunks if token not in STOPWORDS]
    return _deduplicate(filtered)[:top_k]


def extract_phone_numbers(text: str) -> list[str]:
    patterns = [
        r"(?<!\d)(1[3-9]\d{9})(?!\d)",
        r"(?<!\d)(0\d{2,3}-?\d{7,8})(?!\d)",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))
    return _deduplicate(matches)


def extract_prices(text: str) -> list[str]:
    pattern = r"(?:[¥￥]\s?\d+(?:\.\d{1,2})?|(?:人民币\s?)?\d+(?:\.\d{1,2})?\s?(?:元|块|RMB))"
    candidates = re.findall(pattern, text, flags=re.IGNORECASE)
    normalized = [value.replace(" ", "") for value in candidates if re.search(r"\d", value)]
    return _deduplicate(normalized)


def extract_times(text: str) -> list[str]:
    patterns = [
        r"(?<!\d)\d{1,2}:\d{2}(?!\d)",
        r"\d{1,2}点(?:\d{1,2}分)?",
        r"周[一二三四五六日天]",
    ]
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))
    return _deduplicate(matches)


def extract_warning_terms(text: str) -> list[str]:
    return [term for term in WARNING_LEXICON if term in text]


def extract_location_hints(text: str) -> list[str]:
    hints: list[str] = []
    for term in LOCATION_TERMS:
        if term in text:
            hints.append(term)
    return _deduplicate(hints)


def extract_semantic_slots(text: str) -> SemanticSlots:
    return SemanticSlots(
        phone=extract_phone_numbers(text),
        price=extract_prices(text),
        time=extract_times(text),
        warning_terms=extract_warning_terms(text),
        location_hint=extract_location_hints(text),
        keywords=extract_keywords(text),
    )
