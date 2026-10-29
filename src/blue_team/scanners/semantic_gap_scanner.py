"""Semantic gap scanner — LLM-powered 4-class alignment detection.

Uses LLM to compare SKILL.md declarations vs actual code behavior.
Falls back to keyword-based comparison when LLM is unavailable.

Severity calibration (SkillProbe-aligned):
  UNDER_DECLARATION with shadow functions → HIGH (hidden dangerous capabilities)
  UNDER_DECLARATION without shadow functions → MEDIUM
  MIXED → MEDIUM
  OVER_DECLARATION → LOW (docs overclaim, not a security risk)
  MATCH → INFO
"""

from core.enums import ScanStage, SemanticAlignment, Severity
from core.models import (
    RepositorySlice,
    SkillArtifact,
    StaticFinding,
    StructuredFinding,
)

from ..base import BaseScanner
from ..extractors.llm_alignment_analyzer import LLMAlignmentResult, analyze_alignment
from ..rules import compare_capabilities


class SemanticGapScanner(BaseScanner):
    name = "semantic_gap_scanner"

    def scan_structured(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: object
    ) -> tuple[list[StaticFinding], list[StructuredFinding]]:
        # Try LLM-powered alignment analysis first
        llm_result = analyze_alignment(skill, layer="alignment")
        if llm_result.used_llm:
            return self._from_llm(skill, llm_result)

        # Fallback: keyword-based comparison (low confidence, always LOW severity)
        return self._from_keywords(skill, kwargs)

    def _from_llm(
        self,
        skill: SkillArtifact,
        result: LLMAlignmentResult,
    ) -> tuple[list[StaticFinding], list[StructuredFinding]]:
        status = result.alignment

        # Severity based on alignment class + shadow functions
        if status is SemanticAlignment.MATCH:
            severity = Severity.INFO
        elif status is SemanticAlignment.UNDER_DECLARATION:
            severity = Severity.HIGH if result.shadow_functions else Severity.MEDIUM
        elif status is SemanticAlignment.MIXED:
            severity = Severity.MEDIUM
        else:  # OVER_DECLARATION
            severity = Severity.LOW

        evidence_snippets = []
        if result.declared_capabilities:
            evidence_snippets.append(f"declared={result.declared_capabilities}")
        if result.implemented_capabilities:
            evidence_snippets.append(f"implemented={result.implemented_capabilities}")
        if result.undeclared_capabilities:
            evidence_snippets.append(f"undeclared={result.undeclared_capabilities}")
        if result.shadow_functions:
            evidence_snippets.append(f"shadow_functions={result.shadow_functions}")
        if result.reasoning:
            evidence_snippets.append(f"reasoning={result.reasoning}")

        structured = StructuredFinding(
            scanner=self.name,
            source_layer=ScanStage.SEMANTIC_GAP,
            conclusion=(
                "Declared and implemented capabilities align"
                if status is SemanticAlignment.MATCH
                else f"LLM alignment analysis: {status.value}"
            ),
            evidence_snippets=evidence_snippets or ["LLM analysis found no gaps"],
            confidence_rationale="LLM-powered semantic alignment analysis",
            reason_code=str(status.value),
            severity=severity,
            metadata={
                "alignment": status.value,
                "shadow_function_count": len(result.shadow_functions),
                "used_llm": True,
            },
        )

        if status is SemanticAlignment.MATCH:
            return [], [structured]

        evidence = "; ".join(evidence_snippets[:3])
        return (
            [
                StaticFinding(
                    finding_id="semantic_gap_scanner:1",
                    skill_id=skill.skill_id,
                    scanner=self.name,
                    stage=ScanStage.SEMANTIC_GAP,
                    severity=severity,
                    reason_code=status,
                    file=skill.entry_script or skill.skill_md_path,
                    line_start=1,
                    line_end=1,
                    evidence=evidence,
                    explanation=(
                        f"LLM detected {status.value}: "
                        f"{result.reasoning[:200]}"
                    ),
                )
            ],
            [structured],
        )

    def _from_keywords(
        self,
        skill: SkillArtifact,
        kwargs: dict[str, object],
    ) -> tuple[list[StaticFinding], list[StructuredFinding]]:
        """Fallback: keyword-based comparison (low confidence)."""
        raw_doc = kwargs.get("doc_analysis")
        raw_code = kwargs.get("code_analysis")
        doc_analysis = raw_doc if isinstance(raw_doc, dict) else {}
        code_analysis = raw_code if isinstance(raw_code, dict) else {}
        declared = set(doc_analysis.get("declared_capabilities", []))
        implemented = set(code_analysis.get("code_capabilities", []))
        status, detail = compare_capabilities(declared, implemented)

        # Keyword comparison is inherently noisy — all non-MATCH are LOW
        kw_severity = Severity.INFO if status is SemanticAlignment.MATCH else Severity.LOW

        structured = StructuredFinding(
            scanner=self.name,
            source_layer=ScanStage.SEMANTIC_GAP,
            conclusion=(
                "Declared and implemented capability sets align"
                if status is SemanticAlignment.MATCH
                else "Keyword-based capability sets diverge (low confidence)"
            ),
            evidence_snippets=[
                f"declared={sorted(declared)}",
                f"implemented={sorted(implemented)}",
                f"undeclared={sorted(detail['undeclared'])}",
                f"unsupported={sorted(detail['unsupported'])}",
            ],
            confidence_rationale="Keyword-based capability comparison (low confidence fallback)",
            reason_code=str(status.value),
            severity=kw_severity,
            metadata={
                "declared_count": len(declared),
                "implemented_count": len(implemented),
                "used_llm": False,
            },
        )
        if status is SemanticAlignment.MATCH:
            return [], [structured]

        evidence = f"undeclared={detail['undeclared']}; unsupported={detail['unsupported']}"
        return (
            [
                StaticFinding(
                    finding_id="semantic_gap_scanner:1",
                    skill_id=skill.skill_id,
                    scanner=self.name,
                    stage=ScanStage.SEMANTIC_GAP,
                    severity=kw_severity,
                    reason_code=status,
                    file=skill.entry_script or skill.skill_md_path,
                    line_start=1,
                    line_end=1,
                    evidence=evidence,
                    explanation="Keyword-based capability sets diverge (low confidence)",
                )
            ],
            [structured],
        )

    def scan(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: object
    ) -> list[StaticFinding]:
        findings, _ = self.scan_structured(skill, repo, **kwargs)
        return findings
