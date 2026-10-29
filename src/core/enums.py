from __future__ import annotations

from enum import Enum


class SeverityLevel(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingStage(str, Enum):
    REPO_CONTEXT = "repo_context_scan"
    GUIDANCE = "guidance_scan"
    SEMANTIC_GAP = "semantic_gap_scan"
    DYNAMIC_PROBE = "dynamic_probe"
    COMPOSITION = "composition_risk"
    REPORT = "report"


class ContextMatchType(str, Enum):
    MATCH = "context_match"
    PARTIAL = "context_partial_match"
    PARTIAL_MATCH = "context_partial_match"
    CONFLICT = "context_conflict"
    UNKNOWN = "context_unknown"


class SemanticAlignmentType(str, Enum):
    MATCH = "match"
    OVER_DECLARATION = "over_declaration"
    UNDER_DECLARATION = "under_declaration"
    MIXED = "mixed"


class VerdictStatus(str, Enum):
    APPROVED = "approved"
    CONDITIONAL = "conditional"
    REVIEW = "conditional"
    REJECTED = "rejected"


class ProbeKind(str, Enum):
    BENIGN = "benign"
    AMBIGUOUS = "ambiguous"
    EDGE = "edge"


# Backward-compatible aliases used by the current scaffold.
Severity = SeverityLevel
ScanStage = FindingStage
ContextLabel = ContextMatchType
SemanticAlignment = SemanticAlignmentType
