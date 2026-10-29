"""Map scorecards and findings to a final verdict.

SkillProbe-aligned 3-dimensional scoring (§3.3):
  Dim 1 — Malicious Patterns (Gatekeeper: guidance_scan, repo_context_scan)
  Dim 2 — Semantic Consistency (Alignment Detector: semantic_gap_scan)
  Dim 3 — Composition Safety (Flow Simulator: composition_risk)

Verdict rules:
  REJECTED  — any dimension has HIGH/CRITICAL findings
  CONDITIONAL — any dimension has MEDIUM findings, or scores exceed threshold
  APPROVED  — all dimensions clean
"""

from core.enums import FindingStage, SeverityLevel, VerdictStatus
from core.models import AuditScorecard, StaticFinding

_GATEKEEPER_STAGES = {FindingStage.GUIDANCE, FindingStage.REPO_CONTEXT}
_ALIGNMENT_STAGES = {FindingStage.SEMANTIC_GAP}
_COMPOSITION_STAGES = {FindingStage.COMPOSITION, FindingStage.DYNAMIC_PROBE}


class VetoAggregator:
    def choose(
        self, findings: list[StaticFinding], scorecard: AuditScorecard
    ) -> VerdictStatus:
        # Bucket findings into 3 dimensions
        gatekeeper = [f for f in findings if f.stage in _GATEKEEPER_STAGES]
        alignment = [f for f in findings if f.stage in _ALIGNMENT_STAGES]
        composition = [f for f in findings if f.stage in _COMPOSITION_STAGES]

        high_critical = {SeverityLevel.HIGH, SeverityLevel.CRITICAL}

        # Dim 1: Malicious patterns — HIGH/CRITICAL → REJECTED
        if any(f.severity in high_critical for f in gatekeeper):
            return VerdictStatus.REJECTED

        # Dim 2: Semantic consistency — Under-declaration (shadow functions) HIGH → REJECTED
        if any(f.severity in high_critical for f in alignment):
            return VerdictStatus.REJECTED

        # Dim 3: Composition — HIGH/CRITICAL composition edges → REJECTED
        if any(f.severity in high_critical for f in composition):
            return VerdictStatus.REJECTED

        # CONDITIONAL: any dimension has MEDIUM findings or elevated scores
        has_medium = any(
            f.severity == SeverityLevel.MEDIUM
            for f in findings
        )
        has_elevated_score = (
            scorecard.static_risk_score >= 0.6
            or scorecard.dynamic_risk_score >= 0.8
            or scorecard.composition_risk_score >= 0.6
        )

        if has_medium or has_elevated_score:
            return VerdictStatus.CONDITIONAL

        # LOW/INFO only findings — still APPROVED
        # (benign skills with minor composition tags should pass)
        if findings and all(
            f.severity in {SeverityLevel.INFO, SeverityLevel.LOW}
            for f in findings
        ):
            return VerdictStatus.APPROVED

        if findings:
            return VerdictStatus.CONDITIONAL

        return VerdictStatus.APPROVED
