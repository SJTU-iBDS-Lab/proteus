"""Simple multi-axis scoring for blue-team findings."""

from __future__ import annotations

from core.enums import (
    ContextMatchType,
    FindingStage,
    SemanticAlignmentType,
    SeverityLevel,
)
from core.models import (
    AuditScorecard,
    RiskEdge,
    RuntimeTraceSummary,
    SkillArtifact,
    StaticFinding,
)


class RiskScorecard:
    def build(
        self,
        skill: SkillArtifact,
        findings: list[StaticFinding],
        runtime_trace_summary: RuntimeTraceSummary | None,
        composition_risk_edges: list[RiskEdge],
    ) -> AuditScorecard:
        weights = {
            SeverityLevel.INFO: 0.1,
            SeverityLevel.LOW: 0.2,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.HIGH: 1.0,
            SeverityLevel.CRITICAL: 1.5,
        }
        static_score = min(
            1.0, sum(weights[finding.severity] for finding in findings) / 3.0
        )
        dynamic_tag_count = (
            len(runtime_trace_summary.behavioral_risk_tags)
            if runtime_trace_summary is not None
            else 0
        )
        dynamic_score = min(1.0, dynamic_tag_count / 3.0)
        composition_score = min(
            1.0,
            sum(weights[edge.severity] for edge in composition_risk_edges) / 2.0,
        )

        contextual_alignment = ContextMatchType.MATCH
        semantic_alignment = SemanticAlignmentType.MATCH
        for finding in findings:
            if finding.stage is FindingStage.REPO_CONTEXT:
                try:
                    contextual_alignment = ContextMatchType(str(finding.reason_code))
                except ValueError:
                    contextual_alignment = ContextMatchType.CONFLICT
            if finding.stage is FindingStage.SEMANTIC_GAP:
                try:
                    semantic_alignment = SemanticAlignmentType(str(finding.reason_code))
                except ValueError:
                    semantic_alignment = SemanticAlignmentType.MIXED

        severity_breakdown: dict[str, int] = {}
        for finding in findings:
            severity_breakdown[finding.severity.value] = (
                severity_breakdown.get(finding.severity.value, 0) + 1
            )

        return AuditScorecard(
            skill_id=skill.skill_id,
            static_risk_score=round(static_score, 3),
            contextual_alignment=contextual_alignment,
            semantic_alignment=semantic_alignment,
            dynamic_risk_score=round(dynamic_score, 3),
            composition_risk_score=round(composition_score, 3),
            severity_breakdown=severity_breakdown,
        )
