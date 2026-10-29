"""Skill chain scanner — orchestrates AIG scans across a subgraph.

For each skill chain (subgraph from graph_builder), this module:
  1. Runs AIG mcp-scan on every skill in the chain (concurrently)
  2. Aggregates per-skill vulnerabilities
  3. Analyzes whether the combined skill chain is harmful
  4. Provides structured reasons for the verdict
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .aig_scanner import scan_skill_async
from .aig_types import SkillScanResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChainScanResult:
    """Aggregated scan result for a skill chain."""

    subgraph_id: str
    primary_attack_type: str
    hop_length: int
    skill_ids: list[str]

    # Verdict
    is_harmful: bool = False
    confidence: str = "low"  # high / medium / low
    harm_reasons: list[str] = field(default_factory=list)

    # Per-skill results
    skill_results: list[SkillScanResult] = field(default_factory=list)

    # Aggregate metrics
    total_vulns: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    min_score: int = 100
    chain_risk_summary: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "subgraph_id": self.subgraph_id,
            "primary_attack_type": self.primary_attack_type,
            "hop_length": self.hop_length,
            "skill_ids": self.skill_ids,
            "is_harmful": self.is_harmful,
            "confidence": self.confidence,
            "harm_reasons": self.harm_reasons,
            "total_vulns": self.total_vulns,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "min_score": self.min_score,
            "chain_risk_summary": self.chain_risk_summary,
            "skill_results": [r.to_dict() for r in self.skill_results],
            "error": self.error,
        }


def _classify_chain(
    skill_results: list[SkillScanResult],
    attack_type: str,
) -> tuple[bool, str, list[str]]:
    """Determine if a chain is harmful based on aggregated AIG findings.

    Returns:
        (is_harmful, confidence, reasons)
    """
    reasons: list[str] = []

    # Count severities across all skills
    critical = 0
    high = 0
    medium = 0
    harmful_skills: list[str] = []

    for result in skill_results:
        if result.error:
            continue
        for vuln in result.vulnerabilities:
            level = vuln.level.lower()
            if level == "critical":
                critical += 1
                harmful_skills.append(result.skill_id)
                reasons.append(
                    f"[{result.skill_id}] CRITICAL: {vuln.title} — {vuln.risk_type}"
                )
            elif level == "high":
                high += 1
                harmful_skills.append(result.skill_id)
                reasons.append(
                    f"[{result.skill_id}] HIGH: {vuln.title} — {vuln.risk_type}"
                )
            elif level == "medium":
                medium += 1
                reasons.append(
                    f"[{result.skill_id}] MEDIUM: {vuln.title} — {vuln.risk_type}"
                )

    harmful_skills = sorted(set(harmful_skills))

    # Decision logic
    if critical > 0:
        return True, "high", reasons
    if high >= 2:
        return True, "high", reasons
    if high == 1 and medium >= 1:
        return True, "medium", reasons
    if high == 1:
        # Single high-severity vuln — chain attack type matters
        if attack_type in (
            "data_exfiltration",
            "remote_code_execution",
            "command_injection",
            "credential_abuse",
        ):
            return True, "medium", reasons
        return True, "low", reasons
    if medium >= 3:
        return True, "low", reasons

    # No significant findings
    return False, "low", reasons


async def scan_chain_async(
    subgraph_id: str,
    skill_ids: list[str],
    skills_root: str | Path,
    *,
    primary_attack_type: str = "unknown",
    hop_length: int = 0,
    language: str = "en",
    max_concurrent: int = 3,
) -> ChainScanResult:
    """Scan all skills in a chain concurrently with AIG.

    Args:
        subgraph_id: Subgraph identifier.
        skill_ids: Ordered list of skill IDs in the chain.
        skills_root: Root directory containing skill subdirectories.
        primary_attack_type: Expected attack type from graph builder.
        hop_length: Number of hops in the chain.
        language: AIG output language.
        max_concurrent: Max concurrent AIG scans.

    Returns:
        ChainScanResult with aggregated findings and verdict.
    """
    skills_root = Path(skills_root)

    # Run AIG on each skill concurrently (bounded)
    sem = asyncio.Semaphore(max_concurrent)

    async def _bounded_scan(sid: str) -> SkillScanResult:
        async with sem:
            skill_path = skills_root / sid
            return await scan_skill_async(sid, str(skill_path), language=language)

    tasks = [_bounded_scan(sid) for sid in skill_ids]
    skill_results = await asyncio.gather(*tasks)

    # Aggregate
    total_vulns = sum(r.vuln_count for r in skill_results)
    critical = sum(
        1
        for r in skill_results
        for v in r.vulnerabilities
        if v.level.lower() == "critical"
    )
    high = sum(
        1
        for r in skill_results
        for v in r.vulnerabilities
        if v.level.lower() == "high"
    )
    medium = sum(
        1
        for r in skill_results
        for v in r.vulnerabilities
        if v.level.lower() == "medium"
    )
    min_score = min((r.score for r in skill_results if not r.error), default=100)

    # Classify chain
    is_harmful, confidence, reasons = _classify_chain(
        list(skill_results), primary_attack_type
    )

    # Build summary
    summary_parts = [
        f"Chain {subgraph_id}: {len(skill_ids)} skills, {hop_length}-hop",
        f"Attack type: {primary_attack_type}",
        f"Vulns: {critical}C/{high}H/{medium}M (total {total_vulns})",
        f"Min safety score: {min_score}",
        f"Verdict: {'HARMFUL' if is_harmful else 'BENIGN'} ({confidence})",
    ]

    logger.info(" | ".join(summary_parts))

    return ChainScanResult(
        subgraph_id=subgraph_id,
        primary_attack_type=primary_attack_type,
        hop_length=hop_length,
        skill_ids=skill_ids,
        is_harmful=is_harmful,
        confidence=confidence,
        harm_reasons=reasons,
        skill_results=list(skill_results),
        total_vulns=total_vulns,
        critical_count=critical,
        high_count=high,
        medium_count=medium,
        min_score=min_score,
        chain_risk_summary=" | ".join(summary_parts),
    )


def scan_chain(
    subgraph_id: str,
    skill_ids: list[str],
    skills_root: str | Path,
    *,
    primary_attack_type: str = "unknown",
    hop_length: int = 0,
    language: str = "en",
    max_concurrent: int = 3,
) -> ChainScanResult:
    """Synchronous wrapper for scan_chain_async."""
    return asyncio.run(
        scan_chain_async(
            subgraph_id,
            skill_ids,
            skills_root,
            primary_attack_type=primary_attack_type,
            hop_length=hop_length,
            language=language,
            max_concurrent=max_concurrent,
        )
    )
