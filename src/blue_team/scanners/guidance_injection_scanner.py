"""Guidance-aware scanner over SKILL.md lines."""

from core.enums import ScanStage, Severity
from core.models import (
    RepositorySlice,
    SkillArtifact,
    StaticFinding,
    StructuredFinding,
)

from ..base import BaseScanner
from ..rules import GuidanceRuleSet


def _coerce_line_number(value: object) -> int:
    return value if isinstance(value, int) else int(str(value))


class GuidanceInjectionScanner(BaseScanner):
    name = "guidance_injection_scanner"

    def __init__(self) -> None:
        self._rules = GuidanceRuleSet()

    def scan_structured(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: object
    ) -> tuple[list[StaticFinding], list[StructuredFinding]]:
        raw_doc_analysis = kwargs.get("doc_analysis")
        doc_analysis = raw_doc_analysis if isinstance(raw_doc_analysis, dict) else {}
        findings: list[StaticFinding] = []
        structured_findings: list[StructuredFinding] = []
        raw_lines = doc_analysis.get("lines", [])
        lines = (
            [line for line in raw_lines if isinstance(line, str)]
            if isinstance(raw_lines, list)
            else []
        )
        for index, match in enumerate(self._rules.match(lines), start=1):
            typed_severity = (
                match["severity"]
                if isinstance(match["severity"], Severity)
                else Severity.INFO
            )
            line_start = _coerce_line_number(match["line_start"])
            line_end = _coerce_line_number(match["line_end"])
            findings.append(
                StaticFinding(
                    finding_id=f"guidance_injection_scanner:{index}",
                    skill_id=skill.skill_id,
                    scanner=self.name,
                    stage=ScanStage.GUIDANCE,
                    severity=typed_severity,
                    reason_code=str(match["reason_code"]),
                    file=skill.skill_md_path,
                    line_start=line_start,
                    line_end=line_end,
                    evidence=str(match["evidence"]),
                    explanation=str(match["explanation"]),
                )
            )
            structured_findings.append(
                StructuredFinding(
                    scanner=self.name,
                    source_layer=ScanStage.GUIDANCE,
                    conclusion=str(match["explanation"]),
                    evidence_snippets=[str(match["evidence"])],
                    confidence_rationale=(
                        "Deterministic guidance-pattern match over SKILL.md lines "
                        f"at line {line_start}"
                    ),
                    reason_code=str(match["reason_code"]),
                    severity=typed_severity,
                    metadata={
                        "line_start": line_start,
                        "line_end": line_end,
                    },
                )
            )
        if structured_findings:
            return findings, structured_findings
        return findings, [
            StructuredFinding(
                scanner=self.name,
                source_layer=ScanStage.GUIDANCE,
                conclusion="No risky guidance cues detected in SKILL.md",
                evidence_snippets=[
                    "No configured guidance patterns matched any SKILL.md line"
                ],
                confidence_rationale=(
                    f"Deterministic guidance-pattern scan checked {len(lines)} lines and found 0 matches"
                ),
                reason_code="no_guidance_risk",
            )
        ]

    def scan(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: object
    ) -> list[StaticFinding]:
        findings, _ = self.scan_structured(skill, repo, **kwargs)
        return findings
