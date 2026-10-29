from __future__ import annotations

from core.models import RepositorySlice, SkillArtifact

from .finding_schema import finding_from_record, finding_to_record
from .repo_schema import repo_from_record, repo_to_record
from .skill_schema import skill_from_record, skill_to_record
from .trace_schema import (
    trace_from_record,
    trace_summary_from_record,
    trace_summary_to_record,
    trace_to_record,
)
from .subgraph_schema import (
    risk_graph_edge_from_record,
    risk_graph_edge_to_record,
    risk_graph_node_from_record,
    risk_graph_node_to_record,
    skill_fingerprint_from_record,
    skill_fingerprint_to_record,
    subgraph_manifest_from_record,
    subgraph_manifest_to_record,
    validate_subgraph_manifest,
)


def validate_skill_artifact(skill: SkillArtifact) -> SkillArtifact:
    if not skill.skill_id:
        raise ValueError("skill_id is required")
    if not skill.skill_doc_path or not skill.skill_doc:
        raise ValueError("skill document is required")
    return skill


def validate_repository_slice(repo: RepositorySlice) -> RepositorySlice:
    if not repo.repo_id:
        raise ValueError("repo_id is required")
    if not repo.repo_root:
        raise ValueError("repo_root is required")
    return repo


__all__ = [
    "finding_from_record",
    "finding_to_record",
    "repo_from_record",
    "repo_to_record",
    "risk_graph_edge_from_record",
    "risk_graph_edge_to_record",
    "risk_graph_node_from_record",
    "risk_graph_node_to_record",
    "skill_from_record",
    "skill_fingerprint_from_record",
    "skill_fingerprint_to_record",
    "skill_to_record",
    "subgraph_manifest_from_record",
    "subgraph_manifest_to_record",
    "trace_from_record",
    "trace_summary_from_record",
    "trace_summary_to_record",
    "trace_to_record",
    "validate_repository_slice",
    "validate_subgraph_manifest",
    "validate_skill_artifact",
]
