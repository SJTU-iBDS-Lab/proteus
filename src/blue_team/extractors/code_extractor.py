"""Extract lightweight code markers without executing code."""

from core.constants import SENSITIVE_PATH_HINTS
from core.models import SkillArtifact

from ..base import BaseExtractor
from .capability_marker import CapabilityMarker


class CodeExtractor(BaseExtractor):
    def __init__(self) -> None:
        self._marker = CapabilityMarker()

    def extract(self, artifact: SkillArtifact, **kwargs: object) -> dict[str, object]:
        text = artifact.code_text
        lowered = text.lower()
        imports = [
            line.strip()
            for line in text.splitlines()
            if line.startswith("import ") or line.startswith("from ")
        ]
        sensitive_hits = [hint for hint in SENSITIVE_PATH_HINTS if hint in lowered]
        return {
            "imports": imports,
            "code_capabilities": self._marker.mark(text),
            "sensitive_path_hints": sensitive_hits,
            "network_markers": [
                token
                for token in ("requests", "urllib", "http://", "https://")
                if token in lowered
            ],
            "line_count": len(text.splitlines()),
        }
