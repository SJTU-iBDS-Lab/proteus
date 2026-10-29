"""Capability marker that projects text into a shared vocabulary."""

from core.constants import CAPABILITY_KEYWORDS
from shared.utils import normalize_text


class CapabilityMarker:
    def mark(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        capabilities = [
            capability
            for capability, keywords in CAPABILITY_KEYWORDS.items()
            if any(keyword in normalized for keyword in keywords)
        ]
        return sorted(set(capabilities))
