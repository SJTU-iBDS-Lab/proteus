"""Layer E precheck for source/sink composition risk with LLM verification.

Phase 1: Rule-based tag co-occurrence detection (fast, coarse).
Phase 2: LLM semantic verification of flagged edges (SkillProbe P3 Flow Simulator).
"""

from __future__ import annotations

import logging
from typing import override

from core.enums import FindingStage, SeverityLevel
from core.models import (
    RepositorySlice,
    RiskEdge,
    RuntimeTraceSummary,
    SkillArtifact,
    StaticFinding,
)

from ..base import BaseScanner
from ..llm_client import chat_json, is_available

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = """\
You are a security auditor performing composition risk verification.

A rule-based scanner flagged this skill for a potential attack chain:
  Risk type: {risk_type}
  Source tag: {source_tag} (skill can access this data)
  Sink tag: {sink_tag} (skill can send data here)

Your job: determine whether this tag combination represents an ACTUAL security risk
in the context of this specific skill's code, or is a FALSE POSITIVE.

## Common False Positives:
- Skill reads its OWN API key from env/config and calls its OWN API → NOT data exfiltration
- Skill reads its own config files and writes its own output files → NOT persistence attack
- Skill uses subprocess to run its own bundled tools → NOT privilege escalation
- Skill calls a well-known public API (OpenAI, Stripe, etc.) with its own key → normal behavior

## Actual Risks:
- Skill reads ~/.ssh, ~/.aws, /etc/passwd or OTHER users' secrets → real credential access
- Skill sends user files or secrets to an UNKNOWN or ATTACKER-CONTROLLED server → real exfiltration
- Skill executes ARBITRARY user-provided commands via shell → real privilege escalation
- Skill modifies system config (.bashrc, crontab, startup scripts) → real persistence

Analyze the skill's SKILL.md and code below. Answer strictly in JSON:
{{
  "is_harmful": true/false,
  "confidence": "high"/"medium"/"low",
  "reasoning": "one sentence explanation"
}}

## Skill Context:
{skill_context}"""


def _build_skill_context(skill: SkillArtifact) -> str:
    """Build truncated skill context for LLM verification."""
    parts = [f"## SKILL.md\n{skill.skill_doc[:2000]}"]
    code_budget = 4000
    for path, content in skill.code_files.items():
        if code_budget <= 0:
            break
        snippet = content[:code_budget]
        parts.append(f"## {path}\n```\n{snippet}\n```")
        code_budget -= len(snippet)
    return "\n\n".join(parts)


class ChainRiskPrecheckScanner(BaseScanner):
    name: str = "chain_risk_precheck_scanner"

    def __init__(self) -> None:
        self.last_edges: list[RiskEdge] = []

    @override
    def scan(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: object
    ) -> list[StaticFinding]:
        runtime_trace_summary = kwargs.get("runtime_trace_summary")
        trace_summary = (
            runtime_trace_summary
            if isinstance(runtime_trace_summary, RuntimeTraceSummary)
            else RuntimeTraceSummary(skill_id=skill.skill_id)
        )

        source_tags = set(skill.source_tags)
        sink_tags = set(skill.sink_tags)
        if "network_egress_tendency" in trace_summary.behavioral_risk_tags:
            sink_tags.add("network_egress")
        if "sensitive_path_interest" in trace_summary.behavioral_risk_tags:
            source_tags.add("credentials")

        # SkillProbe Table 1 aligned risk link policies.
        # ALL rules require behavioral evidence for HIGH severity.
        # Without behavioral signals, tag co-occurrence alone is MEDIUM at most
        # (many legitimate skills read API keys + call networks).
        has_behavioral = bool(trace_summary.behavioral_risk_tags)
        rule_specs = [
            # P6: credentials + network → data exfiltration
            (
                "credentials",
                "network_egress",
                "data_exfiltration",
                SeverityLevel.HIGH if has_behavioral else SeverityLevel.MEDIUM,
            ),
            # P5/repo: local_files + network → repo exfiltration
            (
                "local_files",
                "network_egress",
                "repo_exfiltration",
                SeverityLevel.MEDIUM if has_behavioral else SeverityLevel.LOW,
            ),
            # persistence: local_files + write → persistence_write
            (
                "local_files",
                "filesystem_write",
                "persistence_write",
                SeverityLevel.MEDIUM if has_behavioral else SeverityLevel.LOW,
            ),
            # P5: credentials + shell → privilege escalation
            (
                "credentials",
                "shell_execution",
                "privilege_escalation",
                SeverityLevel.HIGH if has_behavioral else SeverityLevel.MEDIUM,
            ),
        ]

        edges: list[RiskEdge] = []
        findings: list[StaticFinding] = []
        for index, (source_tag, sink_tag, risk_type, severity) in enumerate(
            rule_specs, start=1
        ):
            if source_tag not in source_tags or sink_tag not in sink_tags:
                continue
            edge = RiskEdge(
                edge_id=f"{skill.skill_id}:{source_tag}->{sink_tag}",
                source_id=f"skill:{skill.skill_id}:{source_tag}",
                sink_id=f"skill:{skill.skill_id}:{sink_tag}",
                risk_type=risk_type,
                severity=severity,
                evidence=(
                    f"source_tag={source_tag}; sink_tag={sink_tag}; "
                    f"behavioral_tags={','.join(trace_summary.behavioral_risk_tags)}"
                ),
                explanation=(
                    "Detected a source/sink pairing that could support a multi-step "
                    "attack chain under Layer E composition precheck."
                ),
                metadata={"repo_id": repo.repo_id},
            )
            edges.append(edge)
            findings.append(
                StaticFinding(
                    finding_id=f"{self.name}:{index}",
                    skill_id=skill.skill_id,
                    scanner=self.name,
                    stage=FindingStage.COMPOSITION,
                    severity=severity,
                    reason_code=risk_type,
                    file=skill.skill_doc_path,
                    line_start=1,
                    line_end=1,
                    evidence=edge.evidence,
                    explanation=edge.explanation,
                    metadata={"edge_id": edge.edge_id},
                )
            )

        # Phase 2: LLM semantic verification (SkillProbe P3 — Flow Simulator)
        # Only verify HIGH+ findings — MEDIUM represents "suspicious but unconfirmed"
        # which is the correct semantic for CONDITIONAL verdict.
        high_plus = {SeverityLevel.HIGH, SeverityLevel.CRITICAL}
        has_high = any(f.severity in high_plus for f in findings)
        if has_high and is_available("flow_sim"):
            findings = self._llm_verify_findings(skill, findings, edges)

        self.last_edges = edges
        return findings

    def _llm_verify_findings(
        self,
        skill: SkillArtifact,
        findings: list[StaticFinding],
        edges: list[RiskEdge],
    ) -> list[StaticFinding]:
        """Use LLM to verify each composition finding and downgrade false positives."""
        skill_context = _build_skill_context(skill)
        verified: list[StaticFinding] = []

        for finding, edge in zip(findings, edges):
            prompt = _VERIFY_PROMPT.format(
                risk_type=edge.risk_type,
                source_tag=finding.evidence.split(";")[0].split("=")[1] if "=" in finding.evidence else "",
                sink_tag=finding.evidence.split(";")[1].split("=")[1].strip() if finding.evidence.count(";") >= 1 else "",
                skill_context=skill_context,
            )
            messages = [{"role": "user", "content": prompt}]
            try:
                result = chat_json(messages, layer="flow_sim", temperature=0.1, timeout=60.0)
            except Exception as exc:
                logger.warning(
                    "LLM verification failed for %s edge %s: %s",
                    skill.skill_id, edge.edge_id, exc,
                )
                verified.append(finding)
                continue

            is_harmful = result.get("is_harmful", True)
            confidence = result.get("confidence", "low")
            reasoning = result.get("reasoning", "")

            if not is_harmful and confidence == "high":
                # LLM says false positive with high confidence → downgrade one level
                downgrade_map = {
                    SeverityLevel.CRITICAL: SeverityLevel.HIGH,
                    SeverityLevel.HIGH: SeverityLevel.MEDIUM,
                    SeverityLevel.MEDIUM: SeverityLevel.LOW,
                    SeverityLevel.LOW: SeverityLevel.LOW,
                    SeverityLevel.INFO: SeverityLevel.INFO,
                }
                new_severity = downgrade_map[finding.severity]
                logger.info(
                    "LLM downgraded %s edge %s (%s→%s): %s",
                    skill.skill_id, edge.edge_id,
                    finding.severity.value, new_severity.value, reasoning,
                )
                verified.append(
                    StaticFinding(
                        finding_id=finding.finding_id,
                        skill_id=finding.skill_id,
                        scanner=finding.scanner,
                        stage=finding.stage,
                        severity=new_severity,
                        reason_code=finding.reason_code,
                        file=finding.file,
                        line_start=finding.line_start,
                        line_end=finding.line_end,
                        evidence=finding.evidence,
                        explanation=f"LLM verified as likely benign (downgraded): {reasoning}",
                        metadata=finding.metadata,
                    )
                )
                edge.severity = new_severity
            else:
                verified.append(finding)

        return verified
