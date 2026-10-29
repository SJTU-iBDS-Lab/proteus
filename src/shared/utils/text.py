"""Small deterministic text helpers."""

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_]{3,}", normalize_text(text))


def collect_keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    normalized = normalize_text(text)
    return [keyword for keyword in keywords if keyword in normalized]
