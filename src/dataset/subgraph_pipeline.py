from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from core.models import (
    RiskGraphEdgeRecord,
    RiskGraphNodeRecord,
    SkillFingerprintRecord,
    SubgraphCoverageStats,
    SubgraphEdgeRef,
    SubgraphManifestRecord,
    SubgraphNodeRef,
    SubgraphPrunedCandidateRecord,
)

from .subgraph_io import (
    ensure_directory,
    write_json,
    write_jsonl_records,
    write_markdown,
)

SCHEMA_VERSION = "subgraph.v1"

RULE_SPECS: list[tuple[str, str, str]] = [
    # P1: user_input → shell_execution = command_injection
    ("user_input", "shell_execution", "command_injection"),
    # P5: external_code → shell_execution = remote_code_execution
    ("external_code", "shell_execution", "remote_code_execution"),
    # P5 variant: external_code → code_execution
    ("external_code", "code_execution", "remote_code_execution"),
    # P6: credentials → network_egress = data_exfiltration
    ("credentials", "network_egress", "data_exfiltration"),
    # P10: credentials → filesystem_write = credential_persistence
    ("credentials", "filesystem_write", "credential_persistence"),
    # P11: local_files → network_egress = repo_exfiltration
    ("local_files", "network_egress", "repo_exfiltration"),
    # P12: local_files → shell_execution = privilege_escalation
    ("local_files", "shell_execution", "privilege_escalation"),
    # P8-adjacent: local_files → sql_execution = data_poisoning
    ("local_files", "sql_execution", "data_poisoning"),
    # user_input → code_execution = code_injection
    ("user_input", "code_execution", "code_injection"),
    # user_input → sql_execution = sql_injection
    ("user_input", "sql_execution", "sql_injection"),
    # credentials → shell_execution = credential_abuse
    ("credentials", "shell_execution", "credential_abuse"),
    # local_files → filesystem_write = persistence_write
    ("local_files", "filesystem_write", "persistence_write"),
    # network_input → shell_execution = remote_command_injection
    ("network_input", "shell_execution", "remote_command_injection"),
    # network_input → code_execution = remote_code_injection
    ("network_input", "code_execution", "remote_code_injection"),
    # network_input → filesystem_write = remote_file_write
    ("network_input", "filesystem_write", "remote_file_write"),
    # network_input → sql_execution = remote_sql_injection
    ("network_input", "sql_execution", "remote_sql_injection"),
]


THEME_SIMILARITY_THRESHOLD = 0.30


def _build_similarity_lookup(
    embeddings: dict[str, list[float]],
    threshold: float,
) -> set[tuple[str, str]]:
    """Precompute which skill pairs pass the theme similarity threshold.

    Returns a set of (skill_a, skill_b) pairs where cosine similarity >= threshold.
    Uses numpy for vectorized computation.
    """
    try:
        import numpy as np
    except ImportError:
        return set()

    skill_ids = sorted(embeddings.keys())
    if not skill_ids:
        return set()

    matrix = np.array([embeddings[sid] for sid in skill_ids], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

    # Compute similarity in chunks to limit memory
    allowed: set[tuple[str, str]] = set()
    chunk_size = 500
    for i in range(0, len(skill_ids), chunk_size):
        chunk = matrix[i : i + chunk_size]
        sim_block = chunk @ matrix.T  # (chunk_size, n)
        rows, cols = np.where(sim_block >= threshold)
        for r, c in zip(rows, cols):
            a = skill_ids[i + int(r)]
            b = skill_ids[int(c)]
            if a != b:
                allowed.add((a, b))

    return allowed


def materialize_nodes(
    records: list[SkillFingerprintRecord],
) -> list[RiskGraphNodeRecord]:
    return [
        RiskGraphNodeRecord(
            node_id=f"node:{record.skill_id}",
            skill_id=record.skill_id,
            cluster_id=record.cluster_id,
            source_tags=list(record.source_tags),
            sink_tags=list(record.sink_tags),
            bridge_tags=list(record.bridge_tags),
            attack_family_candidates=list(record.attack_family_candidates),
            heat_tier=record.heat_tier,
            impl_profile=[record.language, *record.dependency_ecosystem],
            provenance={
                "skill_path": record.skill_path,
                "selection_reason": record.selection_reason,
                "schema_version": record.schema_version,
            },
        )
        for record in records
    ]


def build_edge_candidates(
    nodes: list[RiskGraphNodeRecord],
    *,
    theme_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[RiskGraphEdgeRecord], list[RiskGraphEdgeRecord]]:
    edges: list[RiskGraphEdgeRecord] = []
    pruned: list[RiskGraphEdgeRecord] = []
    seen_pairs: set[tuple[str, str, str]] = set()

    by_source_tag: dict[str, list[RiskGraphNodeRecord]] = {}
    by_sink_tag: dict[str, list[RiskGraphNodeRecord]] = {}
    for node in nodes:
        for tag in node.source_tags:
            by_source_tag.setdefault(tag, []).append(node)
        for tag in node.sink_tags:
            by_sink_tag.setdefault(tag, []).append(node)

    for source_tag, sink_tag, risk_type in RULE_SPECS:
        sources = by_source_tag.get(source_tag, [])
        sinks = by_sink_tag.get(sink_tag, [])
        for source in sources:
            for sink in sinks:
                if source.node_id == sink.node_id:
                    continue
                if theme_pairs is not None and (source.skill_id, sink.skill_id) not in theme_pairs:
                    continue
                dedupe_key = (source.cluster_id, sink.cluster_id, risk_type)
                edge = RiskGraphEdgeRecord(
                    edge_id=(
                        f"edge:{source.skill_id}:{source_tag}"
                        f"->{sink.skill_id}:{sink_tag}"
                    ),
                    source_node_id=source.node_id,
                    sink_node_id=sink.node_id,
                    source_skill_id=source.skill_id,
                    sink_skill_id=sink.skill_id,
                    source_tag=source_tag,
                    sink_tag=sink_tag,
                    risk_type=risk_type,
                    edge_kind="direct",
                    confidence=1.0,
                    support_tags=[source_tag, sink_tag],
                    cluster_pair=[source.cluster_id, sink.cluster_id],
                    evidence_stub=f"rule:{source_tag}->{sink_tag}",
                    selection_reason="rule_match",
                )
                if dedupe_key in seen_pairs:
                    edge.pruned = True
                    edge.prune_reason = "duplicate_cluster_pair"
                    pruned.append(edge)
                    continue
                seen_pairs.add(dedupe_key)
                edges.append(edge)

    return edges, pruned


def extract_direct_subgraphs(
    nodes: list[RiskGraphNodeRecord],
    edges: list[RiskGraphEdgeRecord],
) -> list[SubgraphManifestRecord]:
    nodes_by_id = {node.node_id: node for node in nodes}
    manifests: list[SubgraphManifestRecord] = []
    for edge in edges:
        source = nodes_by_id[edge.source_node_id]
        sink = nodes_by_id[edge.sink_node_id]
        subgraph_id = f"subgraph:direct:{edge.edge_id}"
        manifests.append(
            SubgraphManifestRecord(
                subgraph_id=subgraph_id,
                schema_version=SCHEMA_VERSION,
                subgraph_type="direct",
                primary_attack_type=edge.risk_type,
                hop_length=1,
                nodes=[
                    SubgraphNodeRef(node_id=source.node_id, role="source"),
                    SubgraphNodeRef(node_id=sink.node_id, role="sink"),
                ],
                edges=[SubgraphEdgeRef(edge_id=edge.edge_id, order_index=0)],
                role_signature=["source", "sink"],
                heat_signature=[source.heat_tier, sink.heat_tier],
                difficulty_signature=["explicit"],
                cluster_ids=[source.cluster_id, sink.cluster_id],
                selection_reason=edge.selection_reason,
                lineage={
                    "edge_ids": [edge.edge_id],
                    "source_skill_id": source.skill_id,
                    "sink_skill_id": sink.skill_id,
                },
            )
        )
    return manifests


MAX_BRIDGE_PER_NODE = 20


def extract_bridge_subgraphs(
    nodes: list[RiskGraphNodeRecord],
    edges: list[RiskGraphEdgeRecord],
) -> tuple[list[SubgraphManifestRecord], list[SubgraphPrunedCandidateRecord]]:
    nodes_by_id = {node.node_id: node for node in nodes}
    incoming_by_node: dict[str, list[RiskGraphEdgeRecord]] = {}
    outgoing_by_node: dict[str, list[RiskGraphEdgeRecord]] = {}

    for edge in edges:
        incoming_by_node.setdefault(edge.sink_node_id, []).append(edge)
        outgoing_by_node.setdefault(edge.source_node_id, []).append(edge)

    manifests: list[SubgraphManifestRecord] = []
    pruned_candidates: list[SubgraphPrunedCandidateRecord] = []
    seen_signatures: set[tuple[str, str, str, str, str]] = set()

    for bridge in nodes:
        incoming = incoming_by_node.get(bridge.node_id, [])
        outgoing = outgoing_by_node.get(bridge.node_id, [])
        if not incoming or not outgoing:
            continue
        bridge_count_for_node = 0
        if not bridge.bridge_tags:
            pruned_candidates.append(
                SubgraphPrunedCandidateRecord(
                    candidate_type="bridge",
                    prune_reason="missing_bridge_tag",
                    detail="node has both incoming and outgoing edges but no bridge_tags",
                    bridge_node_id=bridge.node_id,
                    edge_ids=[*{edge.edge_id for edge in [*incoming, *outgoing]}],
                    signature=[bridge.cluster_id],
                )
            )
            continue
        for left_edge in incoming:
            for right_edge in outgoing:
                if left_edge.source_node_id == right_edge.sink_node_id:
                    continue
                source = nodes_by_id[left_edge.source_node_id]
                sink = nodes_by_id[right_edge.sink_node_id]
                signature = (
                    source.cluster_id,
                    bridge.cluster_id,
                    sink.cluster_id,
                    left_edge.risk_type,
                    right_edge.risk_type,
                )
                if signature in seen_signatures:
                    pruned_candidates.append(
                        SubgraphPrunedCandidateRecord(
                            candidate_type="bridge",
                            prune_reason="duplicate_bridge_signature",
                            detail="bridge candidate duplicates an existing cluster-level bridge signature",
                            source_node_id=source.node_id,
                            bridge_node_id=bridge.node_id,
                            sink_node_id=sink.node_id,
                            edge_ids=[left_edge.edge_id, right_edge.edge_id],
                            signature=list(signature),
                        )
                    )
                    continue
                seen_signatures.add(signature)
                if bridge_count_for_node >= MAX_BRIDGE_PER_NODE:
                    continue
                bridge_count_for_node += 1
                manifests.append(
                    SubgraphManifestRecord(
                        subgraph_id=(
                            f"subgraph:bridge:{left_edge.edge_id}|{right_edge.edge_id}"
                        ),
                        schema_version=SCHEMA_VERSION,
                        subgraph_type="bridge",
                        primary_attack_type=right_edge.risk_type,
                        hop_length=2,
                        nodes=[
                            SubgraphNodeRef(node_id=source.node_id, role="source"),
                            SubgraphNodeRef(node_id=bridge.node_id, role="bridge"),
                            SubgraphNodeRef(node_id=sink.node_id, role="sink"),
                        ],
                        edges=[
                            SubgraphEdgeRef(edge_id=left_edge.edge_id, order_index=0),
                            SubgraphEdgeRef(edge_id=right_edge.edge_id, order_index=1),
                        ],
                        role_signature=["source", "bridge", "sink"],
                        heat_signature=[
                            source.heat_tier,
                            bridge.heat_tier,
                            sink.heat_tier,
                        ],
                        difficulty_signature=["explicit"],
                        cluster_ids=[
                            source.cluster_id,
                            bridge.cluster_id,
                            sink.cluster_id,
                        ],
                        selection_reason="bridge_chain_rule_match",
                        lineage={
                            "edge_ids": [left_edge.edge_id, right_edge.edge_id],
                            "source_skill_id": source.skill_id,
                            "bridge_skill_id": bridge.skill_id,
                            "sink_skill_id": sink.skill_id,
                        },
                    )
                )
    return manifests, pruned_candidates


def _is_stateful_node(node: RiskGraphNodeRecord) -> bool:
    """A stateful node writes state AND reads it back, but is NOT a remote relay."""
    return (
        "filesystem_write" in node.sink_tags
        and "local_files" in node.source_tags
        and "network_input" not in node.source_tags
    )


def _is_remote_relay_node(node: RiskGraphNodeRecord) -> bool:
    """A remote relay fetches external data but does NOT read local files as primary source."""
    return (
        "network_input" in node.source_tags
        and bool(node.sink_tags)
        and "local_files" not in node.source_tags
    )


def extract_stateful_subgraphs(
    nodes: list[RiskGraphNodeRecord],
    edges: list[RiskGraphEdgeRecord],
) -> list[SubgraphManifestRecord]:
    """Stateful: source -> persistent_state -> later_trigger.

    The middle node must be a stateful node (writes AND reads local state).
    """
    nodes_by_id = {node.node_id: node for node in nodes}
    incoming_by_node: dict[str, list[RiskGraphEdgeRecord]] = {}
    outgoing_by_node: dict[str, list[RiskGraphEdgeRecord]] = {}

    for edge in edges:
        incoming_by_node.setdefault(edge.sink_node_id, []).append(edge)
        outgoing_by_node.setdefault(edge.source_node_id, []).append(edge)

    manifests: list[SubgraphManifestRecord] = []
    seen: set[tuple[str, str, str]] = set()

    for mid_node in nodes:
        if not _is_stateful_node(mid_node):
            continue
        incoming = incoming_by_node.get(mid_node.node_id, [])
        outgoing = outgoing_by_node.get(mid_node.node_id, [])
        if not incoming or not outgoing:
            continue
        for left in incoming:
            for right in outgoing:
                if left.source_node_id == right.sink_node_id:
                    continue
                source = nodes_by_id[left.source_node_id]
                sink = nodes_by_id[right.sink_node_id]
                key = (source.cluster_id, mid_node.cluster_id, sink.cluster_id)
                if key in seen:
                    continue
                seen.add(key)
                manifests.append(
                    SubgraphManifestRecord(
                        subgraph_id=f"subgraph:stateful:{left.edge_id}|{right.edge_id}",
                        schema_version=SCHEMA_VERSION,
                        subgraph_type="stateful",
                        primary_attack_type=right.risk_type,
                        hop_length=2,
                        nodes=[
                            SubgraphNodeRef(node_id=source.node_id, role="source"),
                            SubgraphNodeRef(node_id=mid_node.node_id, role="persistent_state"),
                            SubgraphNodeRef(node_id=sink.node_id, role="sink"),
                        ],
                        edges=[
                            SubgraphEdgeRef(edge_id=left.edge_id, order_index=0),
                            SubgraphEdgeRef(edge_id=right.edge_id, order_index=1),
                        ],
                        role_signature=["source", "persistent_state", "sink"],
                        heat_signature=[source.heat_tier, mid_node.heat_tier, sink.heat_tier],
                        difficulty_signature=["delayed_trigger"],
                        cluster_ids=[source.cluster_id, mid_node.cluster_id, sink.cluster_id],
                        selection_reason="stateful_chain_match",
                        lineage={
                            "edge_ids": [left.edge_id, right.edge_id],
                            "source_skill_id": source.skill_id,
                            "stateful_skill_id": mid_node.skill_id,
                            "sink_skill_id": sink.skill_id,
                        },
                    )
                )
    return manifests


def extract_remote_drift_subgraphs(
    nodes: list[RiskGraphNodeRecord],
    edges: list[RiskGraphEdgeRecord],
) -> list[SubgraphManifestRecord]:
    """Remote drift: loader -> remote_relay -> sink.

    The middle node fetches remote data (network_input) and has a dangerous sink.
    """
    nodes_by_id = {node.node_id: node for node in nodes}
    incoming_by_node: dict[str, list[RiskGraphEdgeRecord]] = {}
    outgoing_by_node: dict[str, list[RiskGraphEdgeRecord]] = {}

    for edge in edges:
        incoming_by_node.setdefault(edge.sink_node_id, []).append(edge)
        outgoing_by_node.setdefault(edge.source_node_id, []).append(edge)

    manifests: list[SubgraphManifestRecord] = []
    seen: set[tuple[str, str, str]] = set()

    for relay in nodes:
        if not _is_remote_relay_node(relay):
            continue
        incoming = incoming_by_node.get(relay.node_id, [])
        outgoing = outgoing_by_node.get(relay.node_id, [])
        if not incoming or not outgoing:
            continue
        for left in incoming:
            for right in outgoing:
                if left.source_node_id == right.sink_node_id:
                    continue
                source = nodes_by_id[left.source_node_id]
                sink = nodes_by_id[right.sink_node_id]
                key = (source.cluster_id, relay.cluster_id, sink.cluster_id)
                if key in seen:
                    continue
                seen.add(key)
                manifests.append(
                    SubgraphManifestRecord(
                        subgraph_id=f"subgraph:remote_drift:{left.edge_id}|{right.edge_id}",
                        schema_version=SCHEMA_VERSION,
                        subgraph_type="remote_drift",
                        primary_attack_type=right.risk_type,
                        hop_length=2,
                        nodes=[
                            SubgraphNodeRef(node_id=source.node_id, role="source"),
                            SubgraphNodeRef(node_id=relay.node_id, role="remote_relay"),
                            SubgraphNodeRef(node_id=sink.node_id, role="sink"),
                        ],
                        edges=[
                            SubgraphEdgeRef(edge_id=left.edge_id, order_index=0),
                            SubgraphEdgeRef(edge_id=right.edge_id, order_index=1),
                        ],
                        role_signature=["source", "remote_relay", "sink"],
                        heat_signature=[source.heat_tier, relay.heat_tier, sink.heat_tier],
                        difficulty_signature=["remote_update"],
                        cluster_ids=[source.cluster_id, relay.cluster_id, sink.cluster_id],
                        selection_reason="remote_drift_chain_match",
                        lineage={
                            "edge_ids": [left.edge_id, right.edge_id],
                            "source_skill_id": source.skill_id,
                            "relay_skill_id": relay.skill_id,
                            "sink_skill_id": sink.skill_id,
                        },
                    )
                )
    return manifests


def derive_bridge_pruning_from_pruned_edges(
    nodes: list[RiskGraphNodeRecord],
    pruned_edges: list[RiskGraphEdgeRecord],
) -> list[SubgraphPrunedCandidateRecord]:
    nodes_by_id = {node.node_id: node for node in nodes}
    candidates: list[SubgraphPrunedCandidateRecord] = []
    for edge in pruned_edges:
        source = nodes_by_id.get(edge.source_node_id)
        sink = nodes_by_id.get(edge.sink_node_id)
        if source is None or sink is None:
            continue
        if not (source.bridge_tags or sink.bridge_tags):
            continue
        candidates.append(
            SubgraphPrunedCandidateRecord(
                candidate_type="bridge",
                prune_reason="duplicate_bridge_signature",
                detail="bridge-related candidate was suppressed by earlier edge-level deduplication",
                source_node_id=edge.source_node_id,
                bridge_node_id=(source.node_id if source.bridge_tags else sink.node_id),
                sink_node_id=edge.sink_node_id,
                edge_ids=[edge.edge_id],
                signature=[*edge.cluster_pair, edge.risk_type],
            )
        )
    return candidates


def build_subgraph_stats(
    nodes: list[RiskGraphNodeRecord],
    edges: list[RiskGraphEdgeRecord],
    subgraphs: list[SubgraphManifestRecord],
) -> SubgraphCoverageStats:
    topology = Counter(subgraph.subgraph_type for subgraph in subgraphs)
    attack_families = Counter(subgraph.primary_attack_type for subgraph in subgraphs)
    roles = Counter(role for subgraph in subgraphs for role in subgraph.role_signature)
    hop_lengths = Counter(str(subgraph.hop_length) for subgraph in subgraphs)
    heat = Counter(
        signature for subgraph in subgraphs for signature in subgraph.heat_signature
    )
    difficulty = Counter(
        signature
        for subgraph in subgraphs
        for signature in subgraph.difficulty_signature
    )
    cluster_ids = {
        cluster_id for subgraph in subgraphs for cluster_id in subgraph.cluster_ids
    }

    return SubgraphCoverageStats(
        total_nodes=len(nodes),
        total_edges=len(edges),
        total_subgraphs=len(subgraphs),
        topology_distribution=dict(topology),
        attack_family_distribution=dict(attack_families),
        role_distribution=dict(roles),
        hop_length_distribution=dict(hop_lengths),
        heat_distribution=dict(heat),
        difficulty_distribution=dict(difficulty),
        cluster_coverage_count=len(cluster_ids),
        schema_version=SCHEMA_VERSION,
    )


def render_subgraph_coverage(stats: SubgraphCoverageStats) -> str:
    return "\n".join(
        [
            "# Subgraph Coverage Summary",
            "",
            f"- total_nodes: {stats.total_nodes}",
            f"- total_edges: {stats.total_edges}",
            f"- total_subgraphs: {stats.total_subgraphs}",
            f"- topology_distribution: {stats.topology_distribution}",
            f"- attack_family_distribution: {stats.attack_family_distribution}",
            f"- role_distribution: {stats.role_distribution}",
            f"- hop_length_distribution: {stats.hop_length_distribution}",
        ]
    )


def select_evaluation_subgraphs(
    candidates: list[SubgraphManifestRecord],
    *,
    max_per_stratum: int = 10,
) -> list[SubgraphManifestRecord]:
    """Coverage-guided selection: keep at most max_per_stratum per (topology, attack_type).

    Within each stratum, prefer diverse clusters and heat tiers.
    """
    strata: dict[tuple[str, str], list[SubgraphManifestRecord]] = {}
    for sg in candidates:
        key = (sg.subgraph_type, sg.primary_attack_type)
        strata.setdefault(key, []).append(sg)

    selected: list[SubgraphManifestRecord] = []
    for _key, group in sorted(strata.items()):
        if len(group) <= max_per_stratum:
            selected.extend(group)
            continue
        # Prefer cluster diversity: pick one per cluster first
        by_cluster: dict[str, SubgraphManifestRecord] = {}
        remainder: list[SubgraphManifestRecord] = []
        for sg in group:
            cluster_key = "|".join(sorted(sg.cluster_ids))
            if cluster_key not in by_cluster:
                by_cluster[cluster_key] = sg
            else:
                remainder.append(sg)
        diverse = list(by_cluster.values())
        if len(diverse) >= max_per_stratum:
            selected.extend(diverse[:max_per_stratum])
        else:
            selected.extend(diverse)
            remaining_budget = max_per_stratum - len(diverse)
            selected.extend(remainder[:remaining_budget])

    return selected


def split_subgraphs(
    subgraphs: list[SubgraphManifestRecord],
    *,
    seed: int = 42,
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
) -> dict[str, list[SubgraphManifestRecord]]:
    """Deterministic train/dev/eval split."""
    indices = list(range(len(subgraphs)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n = len(indices)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    return {
        "train": [subgraphs[i] for i in indices[:train_end]],
        "dev": [subgraphs[i] for i in indices[train_end:dev_end]],
        "eval": [subgraphs[i] for i in indices[dev_end:]],
    }


def build_subgraph_dataset(
    records: list[SkillFingerprintRecord], output_dir: Path
) -> dict[str, Path]:
    output_dir = ensure_directory(output_dir)
    nodes = materialize_nodes(records)
    embeddings = {
        rec.skill_id: rec.doc_embedding
        for rec in records
        if rec.doc_embedding
    }
    theme_pairs: set[tuple[str, str]] | None = None
    if embeddings:
        theme_pairs = _build_similarity_lookup(embeddings, THEME_SIMILARITY_THRESHOLD)
    edges, pruned = build_edge_candidates(nodes, theme_pairs=theme_pairs)
    subgraphs = extract_direct_subgraphs(nodes, edges)
    bridge_subgraphs, bridge_pruned = extract_bridge_subgraphs(nodes, edges)
    subgraphs.extend(bridge_subgraphs)
    subgraphs.extend(extract_stateful_subgraphs(nodes, edges))
    subgraphs.extend(extract_remote_drift_subgraphs(nodes, edges))
    bridge_pruned.extend(derive_bridge_pruning_from_pruned_edges(nodes, pruned))
    selected = select_evaluation_subgraphs(subgraphs)
    splits = split_subgraphs(selected)
    stats = build_subgraph_stats(nodes, edges, subgraphs)

    nodes_path = write_jsonl_records(
        output_dir / "nodes.jsonl", [node.to_dict() for node in nodes]
    )
    edges_path = write_jsonl_records(
        output_dir / "edges.jsonl", [edge.to_dict() for edge in edges]
    )
    manifest_path = write_jsonl_records(
        output_dir / "subgraph_manifest.jsonl",
        [subgraph.to_dict() for subgraph in subgraphs],
    )
    pruned_path = write_jsonl_records(
        output_dir / "pruned_candidates.jsonl",
        [
            *[edge.to_dict() for edge in pruned],
            *[candidate.to_dict() for candidate in bridge_pruned],
        ],
    )
    stats_path = write_json(output_dir / "subgraph_stats.json", stats.to_dict())
    coverage_path = write_markdown(
        output_dir / "subgraph_coverage.md", render_subgraph_coverage(stats)
    )

    splits_dir = ensure_directory(output_dir / "splits")
    splits_train = write_jsonl_records(
        splits_dir / "train.jsonl",
        [sg.to_dict() for sg in splits["train"]],
    )
    splits_dev = write_jsonl_records(
        splits_dir / "dev.jsonl",
        [sg.to_dict() for sg in splits["dev"]],
    )
    splits_eval = write_jsonl_records(
        splits_dir / "eval.jsonl",
        [sg.to_dict() for sg in splits["eval"]],
    )

    return {
        "nodes": nodes_path,
        "edges": edges_path,
        "subgraph_manifest": manifest_path,
        "pruned_candidates": pruned_path,
        "subgraph_stats": stats_path,
        "subgraph_coverage": coverage_path,
        "splits_train": splits_train,
        "splits_dev": splits_dev,
        "splits_eval": splits_eval,
    }
