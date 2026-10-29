"""Paper-aligned finding / audit-output contracts.

Paper §4.1 specifies three outputs per audit:
  1. decorated feedback  — natural-language wrapper with inline
     ``<FINDING id="...">`` tags that the red team reads as side-info;
  2. structured findings JSON  — list of :class:`PaperFinding` records
     for automated evaluation;
  3. runtime trace summary  — compact log of Layer D's probe runs.

The existing ``core.models.StructuredFinding`` predates the paper and is
kept for backward compatibility with legacy scanners. :class:`PaperFinding`
is the canonical record the CLAW pipeline produces, with a converter at
:meth:`from_structured` so old scanners keep working.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


class Layer:
    """Blue-team layer ids (paper §4.1 A-E)."""
    A_ARTIFACT = "A"
    B_REPO_CONTEXT = "B"
    C_GUIDANCE_AND_GAP = "C"
    D_DYNAMIC_PROBE = "D"
    E_COMPOSITIONAL = "E"


@dataclass(slots=True)
class Evidence:
    """Pointer-style evidence for a finding."""
    file: str = ""
    line_range: tuple[int, int] | None = None
    snippet: str = ""
    rule_id: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line_range": list(self.line_range) if self.line_range else None,
            "snippet": self.snippet,
            "rule_id": self.rule_id,
            "extras": self.extras,
        }


@dataclass(slots=True)
class PaperFinding:
    """Per-layer finding matching CLAW paper §4.1.

    The red team consumes a list of these as ``F_t`` (verbal gradient);
    each finding's ``reason_code`` is the indexable feature the red-team
    agent searches its attack log for.
    """
    finding_id: str                  # stable within a run, e.g. "A-0003"
    layer: str                       # Layer.A_ARTIFACT / B / C / D / E
    category: str                    # e.g. "guidance_injection", "over_declaration"
    reason_code: str                 # e.g. "autonomy_encouragement"
    confidence: float                # 0..1
    severity: str = "medium"         # "info"|"low"|"medium"|"high"|"critical"
    evidence: Evidence = field(default_factory=Evidence)
    explanation: str = ""            # short NL summary for decorated output
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = self.evidence.to_dict()
        return payload

    def decorated_feedback_tag(self) -> str:
        """Render this finding as a ``<FINDING>`` xml-like tag.

        Used by the red-team ``ROUND_USER_TEMPLATE`` findings block and by
        the blue team's decorated-feedback output (paper §4.1 output 1).
        The tag includes ``id``, ``layer``, ``code`` (reason_code) and
        ``severity`` attributes; the body is a short single-line
        explanation with the evidence snippet appended when present.
        """
        body = (self.explanation or "").strip().replace("\n", " ")
        snippet = (self.evidence.snippet or "").strip().replace("\n", " ")
        if snippet:
            # Keep the body compact; the snippet may be noisy.
            snippet = snippet[:240]
            body = f"{body} | evidence: {snippet}" if body else f"evidence: {snippet}"
        if not body:
            body = self.category or self.reason_code
        return (
            f'<FINDING id="{self.finding_id}" layer="{self.layer}" '
            f'code="{self.reason_code}" severity="{self.severity}">'
            f"{body}</FINDING>"
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperFinding":
        ev = d.get("evidence") or {}
        lr = ev.get("line_range")
        evidence = Evidence(
            file=ev.get("file", ""),
            line_range=tuple(lr) if lr else None,
            snippet=ev.get("snippet", ""),
            rule_id=ev.get("rule_id", ""),
            extras=ev.get("extras", {}),
        )
        return cls(
            finding_id=d["finding_id"],
            layer=d["layer"],
            category=d["category"],
            reason_code=d["reason_code"],
            confidence=float(d.get("confidence", 0.5)),
            severity=d.get("severity", "medium"),
            evidence=evidence,
            explanation=d.get("explanation", ""),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_structured(
        cls,
        finding: "object",
        layer: str,
        finding_id: str,
        confidence: float | None = None,
    ) -> "PaperFinding":
        """Adapt a legacy ``core.models.StructuredFinding`` record."""
        # Lazy import to avoid cycle.
        from core.models import StructuredFinding
        assert isinstance(finding, StructuredFinding)
        sev = finding.severity.value if hasattr(finding.severity, "value") else str(finding.severity)
        ev_snippet = "\n".join(finding.evidence_snippets[:3]) if finding.evidence_snippets else ""
        return cls(
            finding_id=finding_id,
            layer=layer,
            category=finding.scanner,
            reason_code=finding.reason_code or finding.scanner,
            confidence=(
                confidence if confidence is not None
                else _severity_to_confidence(sev)
            ),
            severity=sev,
            evidence=Evidence(snippet=ev_snippet),
            explanation=finding.conclusion,
            metadata=dict(finding.metadata or {}),
        )


def _severity_to_confidence(severity: str) -> float:
    return {
        "info": 0.2, "low": 0.4, "medium": 0.6,
        "high": 0.8, "critical": 0.95,
    }.get(severity, 0.5)


@dataclass(slots=True)
class RuntimeTrace:
    """Compact summary emitted by Layer D (dynamic probe)."""
    probes_run: int = 0
    tool_calls_observed: list[str] = field(default_factory=list)
    sensitive_resources_touched: list[str] = field(default_factory=list)
    side_effect_summary: str = ""
    raw_trace_path: str = ""       # pointer to full trace JSON on disk

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AuditOutput:
    """Three-tuple returned by the audit pipeline (paper §4.1)."""
    decorated_feedback: str
    findings: list[PaperFinding]
    runtime_trace: RuntimeTrace
    verdict: str = "undecided"                   # approved | conditional | rejected
    critical_reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decorated_feedback": self.decorated_feedback,
            "findings": [f.to_dict() for f in self.findings],
            "runtime_trace": self.runtime_trace.to_dict(),
            "verdict": self.verdict,
            "critical_reason_codes": self.critical_reason_codes,
        }

    def findings_json(self) -> str:
        return json.dumps(
            [f.to_dict() for f in self.findings],
            ensure_ascii=False,
            indent=2,
        )


__all__ = ["AuditOutput", "Evidence", "Layer", "PaperFinding", "RuntimeTrace"]
