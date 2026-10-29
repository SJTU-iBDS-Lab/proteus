"""Extract document-level cues from SKILL.md."""

from core.models import SkillArtifact
from shared.utils import tokenize

from ..base import BaseExtractor
from .capability_marker import CapabilityMarker


class DocExtractor(BaseExtractor):
    def __init__(self) -> None:
        self._marker = CapabilityMarker()

    def extract(self, artifact: SkillArtifact, **kwargs: object) -> dict[str, object]:
        lines = artifact.skill_text.splitlines()
        headings = [line.strip("# ") for line in lines if line.lstrip().startswith("#")]
        return {
            "headings": headings,
            "line_count": len(lines),
            "declared_capabilities": self._marker.mark(artifact.skill_text),
            "guidance_cues": list(artifact.guidance_cues),
            "tokens": tokenize(artifact.skill_text),
            "lines": lines,
        }
