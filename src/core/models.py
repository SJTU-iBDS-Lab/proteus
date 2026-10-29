from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import cast

from .enums import (
    ContextMatchType,
    FindingStage,
    ProbeKind,
    SemanticAlignmentType,
    SeverityLevel,
    VerdictStatus,
)
from .types import MetadataMap, StringMap


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    raw_items = cast(list[object], value)
    items: list[str] = []
    for raw_item in raw_items:
        if isinstance(raw_item, str):
            items.append(raw_item)
    return items


@dataclass(slots=True)
class SkillArtifact:
    skill_id: str
    skill_root: str
    skill_doc_path: str
    entry_scripts: list[str]
    skill_doc: str
    code_files: StringMap
    metadata: MetadataMap = field(default_factory=dict)
    labels: MetadataMap = field(default_factory=dict)
    permissions_declared: list[str] = field(default_factory=list)
    bootstrap_files: list[str] = field(default_factory=list)
    manifest: MetadataMap = field(default_factory=dict)
    capability_summary: str = ""
    code_capability_summary: str = ""
    source_tags: list[str] = field(default_factory=list)
    sink_tags: list[str] = field(default_factory=list)

    @property
    def skill_text(self) -> str:
        return self.skill_doc

    @property
    def skill_md_path(self) -> str:
        return self.skill_doc_path

    @property
    def entry_script(self) -> str | None:
        return self.entry_scripts[0] if self.entry_scripts else None

    @property
    def code_text(self) -> str:
        ordered_paths = self.entry_scripts or list(self.code_files.keys())
        return "\n\n".join(
            self.code_files[path] for path in ordered_paths if path in self.code_files
        )

    @property
    def declared_capabilities(self) -> list[str]:
        items = _string_list(self.metadata.get("declared_capabilities"))
        return items or list(self.permissions_declared)

    @property
    def code_capabilities(self) -> list[str]:
        return _string_list(self.metadata.get("code_capabilities"))

    @property
    def guidance_cues(self) -> list[str]:
        return _string_list(self.metadata.get("guidance_cues"))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class RepositorySlice:
    repo_id: str
    repo_root: str
    readme_path: str | None
    readme_content: str
    file_tree: list[str]
    dependency_files: list[str]
    labels: MetadataMap = field(default_factory=dict)
    metadata: MetadataMap = field(default_factory=dict)
    extracted_themes: list[str] = field(default_factory=list)

    @property
    def readme_text(self) -> str:
        return self.readme_content

    @property
    def file_inventory(self) -> list[str]:
        return self.file_tree

    @property
    def theme_tokens(self) -> list[str]:
        return self.extracted_themes

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class StaticFinding:
    finding_id: str
    skill_id: str
    scanner: str
    stage: FindingStage
    severity: SeverityLevel
    reason_code: str
    file: str
    line_start: int
    line_end: int
    evidence: str
    explanation: str
    metadata: MetadataMap = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "stage": self.stage.value,
            "severity": self.severity.value,
        }


@dataclass(slots=True)
class StructuredFinding:
    scanner: str
    source_layer: FindingStage
    conclusion: str
    evidence_snippets: list[str]
    confidence_rationale: str
    reason_code: str = ""
    severity: SeverityLevel = SeverityLevel.INFO
    metadata: MetadataMap = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "source_layer": self.source_layer.value,
            "severity": self.severity.value,
        }


@dataclass(slots=True)
class StructuredFindings:
    repo_context: list[StructuredFinding] = field(default_factory=list)
    guidance: list[StructuredFinding] = field(default_factory=list)
    semantic_alignment: list[StructuredFinding] = field(default_factory=list)

    def add(self, finding: StructuredFinding) -> None:
        if finding.source_layer == FindingStage.REPO_CONTEXT:
            self.repo_context.append(finding)
            return
        if finding.source_layer == FindingStage.GUIDANCE:
            self.guidance.append(finding)
            return
        if finding.source_layer == FindingStage.SEMANTIC_GAP:
            self.semantic_alignment.append(finding)

    def extend(self, findings: list[StructuredFinding]) -> None:
        for finding in findings:
            self.add(finding)

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_context": [finding.to_dict() for finding in self.repo_context],
            "guidance": [finding.to_dict() for finding in self.guidance],
            "semantic_alignment": [
                finding.to_dict() for finding in self.semantic_alignment
            ],
        }


@dataclass(slots=True)
class DecoratedFeedback:
    skill_id: str
    decorated_files: StringMap
    findings: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MutationCandidate:
    parent_skill_id: str
    mutation_rationale: str
    mutated_skill_doc: str
    mutated_entry_script: str | None = None
    risk_types: list[str] = field(default_factory=list)
    mutation_tags: list[str] = field(default_factory=list)
    based_on_findings: list[str] = field(default_factory=list)
    metadata: MetadataMap = field(default_factory=dict)
    token_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class AttackLogRecord:
    round_id: str
    skill_id: str
    target_model: str
    mutation_rationale: str = ""
    bypassed: bool = False
    objective_retained: bool = False
    evasion_delta: float = 0.0
    findings: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    feedback_reason_codes: list[str] = field(default_factory=list)
    metadata: MetadataMap = field(default_factory=dict)
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ProbeDefinition:
    probe_id: str
    kind: ProbeKind
    prompt: str
    expectation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "kind": self.kind.value,
            "prompt": self.prompt,
            "expectation": self.expectation,
        }


@dataclass(slots=True)
class ExecutionEvent:
    step: int
    tool: str
    args: MetadataMap
    result_summary: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ProbeObservation:
    probe_name: str
    risk_tags: list[str]
    suggested_tools: list[str]
    sensitive_paths: list[str]
    external_targets: list[str]
    summary: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class RuntimeTraceSummary:
    skill_id: str
    summary: str = ""
    observations: list[ProbeObservation] = field(default_factory=list)
    behavioral_risk_tags: list[str] = field(default_factory=list)
    suggested_tools: list[str] = field(default_factory=list)
    sensitive_paths: list[str] = field(default_factory=list)
    external_targets: list[str] = field(default_factory=list)

    @property
    def runtime_trace_summary(self) -> str:
        return self.summary

    def to_dict(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "summary": self.summary,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "behavioral_risk_tags": self.behavioral_risk_tags,
            "suggested_tools": self.suggested_tools,
            "sensitive_paths": self.sensitive_paths,
            "external_targets": self.external_targets,
        }


@dataclass(slots=True)
class ExecutionTrace:
    run_id: str
    round_id: str
    skill_id: str
    events: list[ExecutionEvent] = field(default_factory=list)
    behavioral_risk_tags: list[str] = field(default_factory=list)
    side_effects: list[MetadataMap] = field(default_factory=list)
    metadata: MetadataMap = field(default_factory=dict)
    runtime_trace_summary: RuntimeTraceSummary | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "round_id": self.round_id,
            "skill_id": self.skill_id,
            "events": [event.to_dict() for event in self.events],
            "behavioral_risk_tags": self.behavioral_risk_tags,
            "side_effects": self.side_effects,
            "metadata": self.metadata,
            "runtime_trace_summary": (
                self.runtime_trace_summary.to_dict()
                if self.runtime_trace_summary is not None
                else None
            ),
        }


@dataclass(slots=True)
class RiskEdge:
    edge_id: str
    source_id: str
    sink_id: str
    risk_type: str
    severity: SeverityLevel
    evidence: str
    explanation: str
    metadata: MetadataMap = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "severity": self.severity.value,
        }


@dataclass(slots=True)
class AuditScorecard:
    skill_id: str
    static_risk_score: float
    contextual_alignment: ContextMatchType
    semantic_alignment: SemanticAlignmentType
    dynamic_risk_score: float
    composition_risk_score: float
    severity_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "static_risk_score": self.static_risk_score,
            "contextual_alignment": self.contextual_alignment.value,
            "semantic_alignment": self.semantic_alignment.value,
            "dynamic_risk_score": self.dynamic_risk_score,
            "composition_risk_score": self.composition_risk_score,
            "severity_breakdown": self.severity_breakdown,
        }


@dataclass(slots=True)
class AuditReport:
    skill_id: str
    verdict: VerdictStatus
    findings: list[StaticFinding]
    decorated_feedback: DecoratedFeedback
    runtime_trace_summary: RuntimeTraceSummary
    composition_risk_edges: list[RiskEdge]
    scorecard: AuditScorecard
    structured_findings: StructuredFindings = field(default_factory=StructuredFindings)

    @property
    def execution_trace(self) -> RuntimeTraceSummary:
        return self.runtime_trace_summary

    @property
    def risk_scorecard(self) -> AuditScorecard:
        return self.scorecard

    @property
    def summary(self) -> str:
        return self.runtime_trace_summary.summary

    def to_dict(self) -> dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "verdict": self.verdict.value,
            "findings": [finding.to_dict() for finding in self.findings],
            "decorated_feedback": self.decorated_feedback.to_dict(),
            "runtime_trace_summary": self.runtime_trace_summary.to_dict(),
            "composition_risk_edges": [
                edge.to_dict() for edge in self.composition_risk_edges
            ],
            "scorecard": self.scorecard.to_dict(),
            "structured_findings": self.structured_findings.to_dict(),
        }


@dataclass(slots=True)
class SkillFingerprintRecord:
    skill_id: str
    skill_path: str
    schema_version: str
    cluster_id: str
    category: str
    heat_tier: str
    language: str
    dependency_ecosystem: list[str]
    metadata: MetadataMap = field(default_factory=dict)
    doc_embedding_ref: str | None = None
    doc_embedding: list[float] = field(default_factory=list)
    embedding_model: str = ""
    code_ops: list[str] = field(default_factory=list)
    dependency_profile: list[str] = field(default_factory=list)
    source_tags: list[str] = field(default_factory=list)
    sink_tags: list[str] = field(default_factory=list)
    bridge_tags: list[str] = field(default_factory=list)
    attack_family_candidates: list[str] = field(default_factory=list)
    selection_reason: str = ""
    api_features: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class RiskGraphNodeRecord:
    node_id: str
    skill_id: str
    cluster_id: str
    source_tags: list[str] = field(default_factory=list)
    sink_tags: list[str] = field(default_factory=list)
    bridge_tags: list[str] = field(default_factory=list)
    attack_family_candidates: list[str] = field(default_factory=list)
    heat_tier: str = ""
    impl_profile: list[str] = field(default_factory=list)
    provenance: MetadataMap = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class RiskGraphEdgeRecord:
    edge_id: str
    source_node_id: str
    sink_node_id: str
    source_skill_id: str
    sink_skill_id: str
    source_tag: str
    sink_tag: str
    risk_type: str
    edge_kind: str
    confidence: float
    support_tags: list[str] = field(default_factory=list)
    cluster_pair: list[str] = field(default_factory=list)
    evidence_stub: str = ""
    selection_reason: str = ""
    pruned: bool = False
    prune_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SubgraphNodeRef:
    node_id: str
    role: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SubgraphEdgeRef:
    edge_id: str
    order_index: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SubgraphManifestRecord:
    subgraph_id: str
    schema_version: str
    subgraph_type: str
    primary_attack_type: str
    hop_length: int
    nodes: list[SubgraphNodeRef] = field(default_factory=list)
    edges: list[SubgraphEdgeRef] = field(default_factory=list)
    role_signature: list[str] = field(default_factory=list)
    heat_signature: list[str] = field(default_factory=list)
    difficulty_signature: list[str] = field(default_factory=list)
    cluster_ids: list[str] = field(default_factory=list)
    selection_reason: str = ""
    lineage: MetadataMap = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SubgraphCoverageStats:
    total_nodes: int
    total_edges: int
    total_subgraphs: int
    topology_distribution: dict[str, int] = field(default_factory=dict)
    attack_family_distribution: dict[str, int] = field(default_factory=dict)
    role_distribution: dict[str, int] = field(default_factory=dict)
    hop_length_distribution: dict[str, int] = field(default_factory=dict)
    heat_distribution: dict[str, int] = field(default_factory=dict)
    difficulty_distribution: dict[str, int] = field(default_factory=dict)
    cluster_coverage_count: int = 0
    schema_version: str = "subgraph.v1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SubgraphValidationIssue:
    issue_id: str
    subgraph_id: str | None
    severity: str
    issue_type: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SubgraphValidationReport:
    schema_version: str
    valid: bool
    issue_count: int
    issues: list[SubgraphValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SubgraphPrunedCandidateRecord:
    candidate_type: str
    prune_reason: str
    detail: str
    source_node_id: str | None = None
    bridge_node_id: str | None = None
    sink_node_id: str | None = None
    edge_ids: list[str] = field(default_factory=list)
    signature: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
