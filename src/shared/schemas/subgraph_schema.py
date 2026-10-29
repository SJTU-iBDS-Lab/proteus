from __future__ import annotations

from typing import cast

from core.models import (
    RiskGraphEdgeRecord,
    RiskGraphNodeRecord,
    SkillFingerprintRecord,
    SubgraphEdgeRef,
    SubgraphManifestRecord,
    SubgraphNodeRef,
)


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    raw = cast(dict[object, object], value)
    result: dict[str, object] = {}
    for key, item in raw.items():
        if isinstance(key, str):
            result[key] = item
    return result


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        return []
    return cast(list[object], value)


def _string_list(value: object) -> list[str]:
    items: list[str] = []
    for item in _object_list(value):
        if isinstance(item, str):
            items.append(item)
    return items


def _optional_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) else None


def _float_value(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _int_value(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def skill_fingerprint_to_record(record: SkillFingerprintRecord) -> dict[str, object]:
    return record.to_dict()


def skill_fingerprint_from_record(record: dict[str, object]) -> SkillFingerprintRecord:
    return SkillFingerprintRecord(
        skill_id=str(record["skill_id"]),
        skill_path=str(record["skill_path"]),
        schema_version=str(record["schema_version"]),
        cluster_id=str(record["cluster_id"]),
        category=str(record["category"]),
        heat_tier=str(record["heat_tier"]),
        language=str(record["language"]),
        dependency_ecosystem=_string_list(record.get("dependency_ecosystem")),
        metadata=_object_dict(record.get("metadata")),
        doc_embedding_ref=_optional_str(record.get("doc_embedding_ref")),
        doc_embedding=[
            float(item)
            for item in _object_list(record.get("doc_embedding"))
            if isinstance(item, (int, float))
        ],
        embedding_model=str(record.get("embedding_model", "")),
        code_ops=_string_list(record.get("code_ops")),
        dependency_profile=_string_list(record.get("dependency_profile")),
        source_tags=_string_list(record.get("source_tags")),
        sink_tags=_string_list(record.get("sink_tags")),
        bridge_tags=_string_list(record.get("bridge_tags")),
        attack_family_candidates=_string_list(record.get("attack_family_candidates")),
        selection_reason=str(record.get("selection_reason", "")),
        api_features={
            str(k): _int_value(v)
            for k, v in _object_dict(record.get("api_features")).items()
        },
    )


def risk_graph_node_to_record(record: RiskGraphNodeRecord) -> dict[str, object]:
    return record.to_dict()


def risk_graph_node_from_record(record: dict[str, object]) -> RiskGraphNodeRecord:
    return RiskGraphNodeRecord(
        node_id=str(record["node_id"]),
        skill_id=str(record["skill_id"]),
        cluster_id=str(record["cluster_id"]),
        source_tags=_string_list(record.get("source_tags")),
        sink_tags=_string_list(record.get("sink_tags")),
        bridge_tags=_string_list(record.get("bridge_tags")),
        attack_family_candidates=_string_list(record.get("attack_family_candidates")),
        heat_tier=str(record.get("heat_tier", "")),
        impl_profile=_string_list(record.get("impl_profile")),
        provenance=_object_dict(record.get("provenance")),
    )


def risk_graph_edge_to_record(record: RiskGraphEdgeRecord) -> dict[str, object]:
    return record.to_dict()


def risk_graph_edge_from_record(record: dict[str, object]) -> RiskGraphEdgeRecord:
    return RiskGraphEdgeRecord(
        edge_id=str(record["edge_id"]),
        source_node_id=str(record["source_node_id"]),
        sink_node_id=str(record["sink_node_id"]),
        source_skill_id=str(record["source_skill_id"]),
        sink_skill_id=str(record["sink_skill_id"]),
        source_tag=str(record["source_tag"]),
        sink_tag=str(record["sink_tag"]),
        risk_type=str(record["risk_type"]),
        edge_kind=str(record["edge_kind"]),
        confidence=_float_value(record.get("confidence"), 0.0),
        support_tags=_string_list(record.get("support_tags")),
        cluster_pair=_string_list(record.get("cluster_pair")),
        evidence_stub=str(record.get("evidence_stub", "")),
        selection_reason=str(record.get("selection_reason", "")),
        pruned=bool(record.get("pruned", False)),
        prune_reason=_optional_str(record.get("prune_reason")),
    )


def subgraph_manifest_to_record(record: SubgraphManifestRecord) -> dict[str, object]:
    return record.to_dict()


def subgraph_manifest_from_record(record: dict[str, object]) -> SubgraphManifestRecord:
    manifest = SubgraphManifestRecord(
        subgraph_id=str(record["subgraph_id"]),
        schema_version=str(record["schema_version"]),
        subgraph_type=str(record["subgraph_type"]),
        primary_attack_type=str(record["primary_attack_type"]),
        hop_length=_int_value(record["hop_length"]),
        nodes=[
            SubgraphNodeRef(
                node_id=str(item["node_id"]),
                role=str(item["role"]),
            )
            for item in (
                _object_dict(item) for item in _object_list(record.get("nodes"))
            )
            if item
        ],
        edges=[
            SubgraphEdgeRef(
                edge_id=str(item["edge_id"]),
                order_index=_int_value(item["order_index"]),
            )
            for item in (
                _object_dict(item) for item in _object_list(record.get("edges"))
            )
            if item
        ],
        role_signature=_string_list(record.get("role_signature")),
        heat_signature=_string_list(record.get("heat_signature")),
        difficulty_signature=_string_list(record.get("difficulty_signature")),
        cluster_ids=_string_list(record.get("cluster_ids")),
        selection_reason=str(record.get("selection_reason", "")),
        lineage=_object_dict(record.get("lineage")),
    )
    return validate_subgraph_manifest(manifest)


def validate_subgraph_manifest(
    manifest: SubgraphManifestRecord,
) -> SubgraphManifestRecord:
    if not manifest.subgraph_id:
        raise ValueError("subgraph_id is required")
    if manifest.subgraph_type not in {"direct", "bridge", "stateful", "remote_drift"}:
        raise ValueError("unsupported subgraph_type")
    if not manifest.nodes or not manifest.edges:
        raise ValueError("subgraph must contain nodes and edges")
    if manifest.subgraph_type == "direct":
        if manifest.hop_length != 1:
            raise ValueError("direct subgraph hop_length must be 1")
        if len(manifest.edges) != 1:
            raise ValueError("direct subgraph must contain exactly one edge")
        if set(manifest.role_signature) != {"source", "sink"}:
            raise ValueError("direct subgraph role_signature must be source/sink")
    if manifest.subgraph_type == "bridge":
        if manifest.hop_length != 2:
            raise ValueError("bridge subgraph hop_length must be 2")
        if len(manifest.edges) != 2:
            raise ValueError("bridge subgraph must contain exactly two edges")
    return manifest
