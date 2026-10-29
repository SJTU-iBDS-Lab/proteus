"""Aggregation and markdown reporting for A.I.G scan results."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .aig_analyzer import AIGResult


@dataclass(slots=True)
class AIGSummary:
    scanned: int
    total_findings: int
    by_level: dict[str, int] = field(default_factory=dict)
    by_risk_type: dict[str, int] = field(default_factory=dict)
    skills_with_findings: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "total_findings": self.total_findings,
            "skills_with_findings": self.skills_with_findings,
            "by_level": dict(self.by_level),
            "by_risk_type": dict(self.by_risk_type),
        }


def summarize(results: list[AIGResult]) -> AIGSummary:
    by_level: Counter[str] = Counter()
    by_risk_type: Counter[str] = Counter()
    skills_with_findings = 0
    total_findings = 0
    for result in results:
        if result.findings:
            skills_with_findings += 1
        for finding in result.findings:
            total_findings += 1
            level = finding.level or "unknown"
            risk_type = finding.risk_type or "unknown"
            by_level[level] += 1
            by_risk_type[risk_type] += 1
    return AIGSummary(
        scanned=len(results),
        total_findings=total_findings,
        by_level=dict(by_level.most_common()),
        by_risk_type=dict(by_risk_type.most_common()),
        skills_with_findings=skills_with_findings,
    )


def render_markdown(summary: AIGSummary, error_count: int = 0) -> str:
    lines = [
        "# A.I.G Deep Scan Summary",
        "",
        f"- Skills scanned: {summary.scanned}",
        f"- Skills with findings: {summary.skills_with_findings}",
        f"- Total findings: {summary.total_findings}",
        f"- Errors: {error_count}",
        "",
        "## Findings by level",
        "",
        "| Level | Count |",
        "|-------|-------|",
    ]
    for level, count in summary.by_level.items():
        lines.append(f"| {level} | {count} |")
    lines.extend(
        [
            "",
            "## Findings by risk type",
            "",
            "| Risk type | Count |",
            "|-----------|-------|",
        ]
    )
    for risk_type, count in summary.by_risk_type.items():
        lines.append(f"| {risk_type} | {count} |")
    return "\n".join(lines) + "\n"


def write_results(
    results: list[AIGResult],
    errors: list[dict[str, str]],
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "aig_summary.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    summary = summarize(results)
    md_path = output_dir / "aig_summary.md"
    md_path.write_text(render_markdown(summary, error_count=len(errors)), encoding="utf-8")

    errors_path = output_dir / "aig_errors.jsonl"
    if errors:
        with errors_path.open("w", encoding="utf-8") as handle:
            for entry in errors:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    elif errors_path.exists():
        errors_path.unlink()

    return jsonl_path, md_path, errors_path


__all__ = [
    "AIGSummary",
    "render_markdown",
    "summarize",
    "write_results",
]
