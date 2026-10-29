"""Data types for AIG scan results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class AIGVulnerability:
    """A single vulnerability found by AIG mcp-scan."""

    title: str
    description: str
    risk_type: str
    level: str  # Critical / High / Medium
    suggestion: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class SkillScanResult:
    """AIG scan result for a single skill."""

    skill_id: str
    skill_path: str
    score: int = 100  # AIG safety score (0-100, higher = safer)
    vulnerabilities: list[AIGVulnerability] = field(default_factory=list)
    readme_summary: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0

    @property
    def is_harmful(self) -> bool:
        return any(
            v.level.lower() in ("critical", "high") for v in self.vulnerabilities
        )

    @property
    def vuln_count(self) -> int:
        return len(self.vulnerabilities)

    def to_dict(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "skill_path": self.skill_path,
            "score": self.score,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "readme_summary": self.readme_summary[:500],
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }
