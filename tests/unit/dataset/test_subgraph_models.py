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


def test_skill_fingerprint_record_serializes_candidate_fields() -> None:
    record = SkillFingerprintRecord(
        skill_id="skill-source",
        skill_path="data/skills/source",
        schema_version="subgraph.v1",
        cluster_id="cluster-a",
        category="memory",
        heat_tier="tail",
        language="python",
        dependency_ecosystem=["pypi"],
        metadata={"downloads": 10},
        doc_embedding_ref=None,
        code_ops=["read_env"],
        dependency_profile=["requests"],
        source_tags=["credentials"],
        sink_tags=[],
        bridge_tags=[],
        attack_family_candidates=["P6"],
        selection_reason="long_tail_risk",
    )

    payload = record.to_dict()

    assert payload["skill_id"] == "skill-source"
    assert payload["heat_tier"] == "tail"
    assert payload["source_tags"] == ["credentials"]
    assert payload["attack_family_candidates"] == ["P6"]


def test_risk_graph_edge_record_serializes_prune_fields() -> None:
    edge = RiskGraphEdgeRecord(
        edge_id="edge:source:credentials->sink:network_egress",
        source_node_id="node:skill-source",
        sink_node_id="node:skill-sink",
        source_skill_id="skill-source",
        sink_skill_id="skill-sink",
        source_tag="credentials",
        sink_tag="network_egress",
        risk_type="data_exfiltration",
        edge_kind="direct",
        confidence=1.0,
        support_tags=["credentials", "network_egress"],
        cluster_pair=["cluster-a", "cluster-b"],
        evidence_stub="rule:credentials->network_egress",
        selection_reason="rule_match",
        pruned=True,
        prune_reason="duplicate_cluster_pair",
    )

    payload = edge.to_dict()

    assert payload["edge_kind"] == "direct"
    assert payload["pruned"] is True
    assert payload["prune_reason"] == "duplicate_cluster_pair"


def test_subgraph_manifest_record_serializes_nested_refs() -> None:
    manifest = SubgraphManifestRecord(
        subgraph_id="subgraph:direct:abc123",
        schema_version="subgraph.v1",
        subgraph_type="direct",
        primary_attack_type="data_exfiltration",
        hop_length=1,
        nodes=[
            SubgraphNodeRef(node_id="node:skill-source", role="source"),
            SubgraphNodeRef(node_id="node:skill-sink", role="sink"),
        ],
        edges=[SubgraphEdgeRef(edge_id="edge:1", order_index=0)],
        role_signature=["source", "sink"],
        heat_signature=["tail", "mid"],
        difficulty_signature=["explicit"],
        cluster_ids=["cluster-a", "cluster-b"],
        selection_reason="minimal_direct_chain",
        lineage={"edge_ids": ["edge:1"]},
    )

    payload = manifest.to_dict()
    nodes = cast(list[dict[str, object]], payload["nodes"])
    edges = cast(list[dict[str, object]], payload["edges"])

    assert payload["subgraph_type"] == "direct"
    assert payload["hop_length"] == 1
    assert nodes[0]["role"] == "source"
    assert edges[0]["edge_id"] == "edge:1"
