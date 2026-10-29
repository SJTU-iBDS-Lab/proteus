"""Repo-aware scanner using deterministic token overlap."""

from core.enums import ContextLabel, ScanStage, Severity
from core.models import (
    RepositorySlice,
    SkillArtifact,
    StaticFinding,
    StructuredFinding,
)

from ..base import BaseScanner
from ..rules import assess_repo_context


class RepoContextScanner(BaseScanner):
    name = "repo_context_scanner"

    def scan_structured(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: object
    ) -> tuple[list[StaticFinding], list[StructuredFinding]]:
        raw_doc_analysis = kwargs.get("doc_analysis")
        raw_repo_analysis = kwargs.get("repo_analysis")
        doc_analysis = raw_doc_analysis if isinstance(raw_doc_analysis, dict) else {}
        repo_analysis = raw_repo_analysis if isinstance(raw_repo_analysis, dict) else {}
        skill_tokens = set(doc_analysis.get("tokens", []))
        repo_tokens = set(repo_analysis.get("theme_tokens", []))
        label, ratio, overlap = assess_repo_context(skill_tokens, repo_tokens)

        if label is ContextLabel.MATCH:
            conclusion = "Skill context strongly matches repository context"
            severity = Severity.INFO
        elif label is ContextLabel.PARTIAL_MATCH:
            conclusion = "Skill context partially matches repository context"
            severity = Severity.INFO
        elif label is ContextLabel.UNKNOWN:
            conclusion = (
                "Repository context is insufficient for a confident match decision"
            )
            # Insufficient context is not evidence of malice
            severity = Severity.LOW
        else:
            conclusion = "Skill context conflicts with repository context"
            # Token-based overlap is crude; LOW unless confirmed by other signals
            severity = Severity.LOW

        structured_finding = StructuredFinding(
            scanner=self.name,
            source_layer=ScanStage.REPO_CONTEXT,
            conclusion=conclusion,
            evidence_snippets=overlap[:5]
            or ["no shared repository context tokens detected"],
            confidence_rationale=(
                f"Deterministic token-overlap heuristic produced ratio={ratio:.2f} "
                f"from {len(skill_tokens)} skill tokens and {len(repo_tokens)} repo tokens"
            ),
            reason_code=str(label.value),
            severity=severity,
            metadata={"overlap_ratio": round(ratio, 4), "overlap_count": len(overlap)},
        )

        if label in {ContextLabel.MATCH, ContextLabel.PARTIAL_MATCH}:
            return [], [structured_finding]
        explanation = (
            "No meaningful overlap with repository context"
            if not overlap
            else "Weak overlap with repository context"
        )
        return (
            [
                StaticFinding(
                    finding_id="repo_context_scanner:1",
                    skill_id=skill.skill_id,
                    scanner=self.name,
                    stage=ScanStage.REPO_CONTEXT,
                    severity=severity,
                    reason_code=label,
                    file=skill.skill_md_path,
                    line_start=1,
                    line_end=1,
                    evidence=f"repo_overlap_ratio={ratio:.2f}; shared_tokens={','.join(overlap[:5])}",
                    explanation=explanation,
                )
            ],
            [structured_finding],
        )

    def scan(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: object
    ) -> list[StaticFinding]:
        findings, _ = self.scan_structured(skill, repo, **kwargs)
        return findings
