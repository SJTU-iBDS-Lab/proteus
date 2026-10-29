from __future__ import annotations

import pytest

from core.models import SubgraphEdgeRef, SubgraphManifestRecord, SubgraphNodeRef
from shared.schemas.subgraph_schema import (
    subgraph_manifest_from_record,
    subgraph_manifest_to_record,
    validate_subgraph_manifest,
)


def test_subgraph_manifest_roundtrip_preserves_roles_and_edges() -> None:
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
        heat_signature=["tail"],
        difficulty_signature=["explicit"],
        cluster_ids=["cluster-a", "cluster-b"],
        selection_reason="minimal_direct_chain",
        lineage={"edge_ids": ["edge:1"]},
    )

    record = subgraph_manifest_to_record(manifest)
    restored = subgraph_manifest_from_record(record)

    assert restored.subgraph_id == manifest.subgraph_id
    assert restored.nodes[0].role == "source"
    assert restored.edges[0].edge_id == "edge:1"


def test_validate_subgraph_manifest_rejects_direct_manifest_with_wrong_hop_length() -> (
    None
):
    manifest = SubgraphManifestRecord(
        subgraph_id="subgraph:direct:bad-hop",
        schema_version="subgraph.v1",
        subgraph_type="direct",
        primary_attack_type="data_exfiltration",
        hop_length=2,
        nodes=[
            SubgraphNodeRef(node_id="node:skill-source", role="source"),
            SubgraphNodeRef(node_id="node:skill-sink", role="sink"),
        ],
        edges=[SubgraphEdgeRef(edge_id="edge:1", order_index=0)],
        role_signature=["source", "sink"],
        heat_signature=["tail"],
        difficulty_signature=["explicit"],
        cluster_ids=["cluster-a", "cluster-b"],
        selection_reason="bad_direct_chain",
        lineage={"edge_ids": ["edge:1"]},
    )

    with pytest.raises(ValueError, match="direct subgraph hop_length"):
        validate_subgraph_manifest(manifest)


def test_validate_subgraph_manifest_accepts_bridge_manifest() -> None:
    manifest = SubgraphManifestRecord(
        subgraph_id="subgraph:bridge:good-hop",
        schema_version="subgraph.v1",
        subgraph_type="bridge",
        primary_attack_type="data_exfiltration",
        hop_length=2,
        nodes=[
            SubgraphNodeRef(node_id="node:skill-source", role="source"),
            SubgraphNodeRef(node_id="node:skill-bridge", role="bridge"),
            SubgraphNodeRef(node_id="node:skill-sink", role="sink"),
        ],
        edges=[
            SubgraphEdgeRef(edge_id="edge:1", order_index=0),
            SubgraphEdgeRef(edge_id="edge:2", order_index=1),
        ],
        role_signature=["source", "bridge", "sink"],
        heat_signature=["tail", "mid", "mid"],
        difficulty_signature=["explicit"],
        cluster_ids=["cluster-a", "cluster-b", "cluster-c"],
        selection_reason="minimal_bridge_chain",
        lineage={"edge_ids": ["edge:1", "edge:2"]},
    )

    validated = validate_subgraph_manifest(manifest)

    assert validated.subgraph_type == "bridge"
    assert validated.hop_length == 2
