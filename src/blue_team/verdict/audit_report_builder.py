"""Build the top-level audit report."""

from __future__ import annotations

from core.enums import VerdictStatus
from core.models import (
    AuditReport,
    AuditScorecard,
    DecoratedFeedback,
    RiskEdge,
    RuntimeTraceSummary,
    SkillArtifact,
    StaticFinding,
    StructuredFindings,
)


class AuditReportBuilder:
    def build(
        self,
        skill: SkillArtifact,
        findings: list[StaticFinding],
        decorated_feedback: DecoratedFeedback,
        runtime_trace_summary: RuntimeTraceSummary | None,
        composition_risk_edges: list[RiskEdge],
        risk_scorecard: AuditScorecard,
        verdict: VerdictStatus,
        structured_findings: StructuredFindings | None = None,
    ) -> AuditReport:
        summary = runtime_trace_summary or RuntimeTraceSummary(
            skill_id=skill.skill_id,
            summary=(
                f"tools=none; risk_tags=none; findings={len(findings)}; "
                f"composition_edges={len(composition_risk_edges)}"
            ),
        )
        return AuditReport(
            skill_id=skill.skill_id,
            findings=findings,
            decorated_feedback=decorated_feedback,
            runtime_trace_summary=summary,
            composition_risk_edges=composition_risk_edges,
            scorecard=risk_scorecard,
            verdict=verdict,
            structured_findings=structured_findings or StructuredFindings(),
        )
