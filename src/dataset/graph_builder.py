"""NetworkX-based full graph construction with multi-hop path enumeration.

Replaces the hardcoded 1-hop / 2-hop topology extraction with a general
directed graph where `nx.all_simple_paths` discovers attack chains of
arbitrary length.

Pipeline:
  1. Build DiGraph: nodes = skills, edges = all valid RULE_SPEC matches
  2. Classify entry points (attack chain origins) and exit points (terminals)
  3. Enumerate all simple paths from entry → exit up to ``max_hops``
  4. Prune by embedding similarity, cluster diversity, and sampling budget
  5. Convert paths to SubgraphManifestRecord for downstream eval
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

import networkx as nx  # type: ignore[import-untyped]
import numpy as np

from core.models import (
    RiskGraphEdgeRecord,
    RiskGraphNodeRecord,
    SkillFingerprintRecord,
    SubgraphEdgeRef,
    SubgraphManifestRecord,
    SubgraphNodeRef,
)

from .subgraph_io import ensure_directory, write_json, write_jsonl_records, write_markdown
from .subgraph_pipeline import (
    RULE_SPECS,
    SCHEMA_VERSION,
    build_subgraph_stats,
    materialize_nodes,
    render_subgraph_coverage,
    split_subgraphs,
)

logger = logging.getLogger(__name__)

# ── Entry / exit point classification ──────────────────────────────────

# Entry points: tags that originate attack data (attacker-controlled input)
ENTRY_SOURCE_TAGS = frozenset({
    "user_input",      # direct user-controlled input
    "network_input",   # remote attacker-controlled data
    "external_code",   # untrusted code from external source
    "credentials",     # stolen/harvested credentials
    "local_files",     # local file content (read sensitive data)
})

# Exit points: tags where attack chains terminate (dangerous effects)
EXIT_SINK_TAGS = frozenset({
    "shell_execution",    # command injection / RCE
    "code_execution",     # arbitrary code execution
    "network_egress",     # data exfiltration
    "sql_execution",      # SQL injection
    "filesystem_write",   # persistence / tampering
})


def _is_entry_node(node: RiskGraphNodeRecord) -> bool:
    """A node that can originate an attack chain (has attacker-reachable sources)."""
    return bool(set(node.source_tags) & ENTRY_SOURCE_TAGS)


def _is_exit_node(node: RiskGraphNodeRecord) -> bool:
    """A node that terminates an attack chain (has dangerous sink effects)."""
    return bool(set(node.sink_tags) & EXIT_SINK_TAGS)


# ── Graph construction ─────────────────────────────────────────────────

def build_digraph(
    nodes: list[RiskGraphNodeRecord],
    *,
    theme_pairs: set[tuple[str, str]] | None = None,
) -> nx.DiGraph:
    """Build a NetworkX DiGraph from nodes and all 16 RULE_SPEC edge rules.

    Each node stores its full RiskGraphNodeRecord as attribute.
    Each edge stores source_tag, sink_tag, risk_type, and edge_id.
    Multi-edges between the same pair (different rules) are stored as
    parallel edges in a MultiDiGraph, then collapsed to DiGraph keeping
    the highest-risk edge.
    """
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in nodes:
        G.add_node(
            node.node_id,
            skill_id=node.skill_id,
            cluster_id=node.cluster_id,
            source_tags=node.source_tags,
            sink_tags=node.sink_tags,
            bridge_tags=node.bridge_tags,
            heat_tier=node.heat_tier,
            is_entry=_is_entry_node(node),
            is_exit=_is_exit_node(node),
        )

    # Index nodes by tag for efficient edge generation
    by_source_tag: dict[str, list[RiskGraphNodeRecord]] = {}
    by_sink_tag: dict[str, list[RiskGraphNodeRecord]] = {}
    for node in nodes:
        for tag in node.source_tags:
            by_source_tag.setdefault(tag, []).append(node)
        for tag in node.sink_tags:
            by_sink_tag.setdefault(tag, []).append(node)

    edge_count = 0

    if theme_pairs is not None:
        # Theme-filtered path: iterate over theme_pairs (much smaller than
        # full cross-product) and check each rule's tag match. This is the
        # production path for the full 47k skill graph.
        by_skill: dict[str, RiskGraphNodeRecord] = {
            node.skill_id: node for node in nodes
        }
        logger.info(
            "Theme-filtered edge construction: %d pairs x %d rules",
            len(theme_pairs),
            len(RULE_SPECS),
        )
        processed = 0
        log_every = max(1, len(theme_pairs) // 20)
        for src_id, snk_id in theme_pairs:
            processed += 1
            if processed % log_every == 0:
                logger.info(
                    "  theme-pair progress: %d/%d (%.1f%%) edges_so_far=%d",
                    processed,
                    len(theme_pairs),
                    100 * processed / len(theme_pairs),
                    edge_count,
                )
            if src_id == snk_id:
                continue
            src_node = by_skill.get(src_id)
            snk_node = by_skill.get(snk_id)
            if src_node is None or snk_node is None:
                continue
            src_tags = set(src_node.source_tags)
            snk_tags = set(snk_node.sink_tags)
            if not src_tags or not snk_tags:
                continue
            for source_tag, sink_tag, risk_type in RULE_SPECS:
                if source_tag not in src_tags or sink_tag not in snk_tags:
                    continue
                edge_id = (
                    f"edge:{src_node.skill_id}:{source_tag}"
                    f"->{snk_node.skill_id}:{sink_tag}"
                )
                if G.has_edge(src_node.node_id, snk_node.node_id):
                    edata = G[src_node.node_id][snk_node.node_id]
                    edata.setdefault("risk_types", []).append(risk_type)
                    edata.setdefault("edge_ids", []).append(edge_id)
                    edata.setdefault("tag_pairs", []).append(
                        (source_tag, sink_tag)
                    )
                else:
                    G.add_edge(
                        src_node.node_id,
                        snk_node.node_id,
                        risk_type=risk_type,
                        risk_types=[risk_type],
                        source_tag=source_tag,
                        sink_tag=sink_tag,
                        edge_id=edge_id,
                        edge_ids=[edge_id],
                        tag_pairs=[(source_tag, sink_tag)],
                    )
                    edge_count += 1
    else:
        # No theme filter: original rule-first nested iteration.
        for rule_idx, (source_tag, sink_tag, risk_type) in enumerate(RULE_SPECS):
            sources = by_source_tag.get(source_tag, [])
            sinks = by_sink_tag.get(sink_tag, [])
            logger.info(
                "  Rule %d/%d %s->%s: %d sources x %d sinks",
                rule_idx + 1,
                len(RULE_SPECS),
                source_tag,
                sink_tag,
                len(sources),
                len(sinks),
            )
            for src_node in sources:
                for snk_node in sinks:
                    if src_node.node_id == snk_node.node_id:
                        continue
                    edge_id = (
                        f"edge:{src_node.skill_id}:{source_tag}"
                        f"->{snk_node.skill_id}:{sink_tag}"
                    )
                    if G.has_edge(src_node.node_id, snk_node.node_id):
                        edata = G[src_node.node_id][snk_node.node_id]
                        edata.setdefault("risk_types", []).append(risk_type)
                        edata.setdefault("edge_ids", []).append(edge_id)
                        edata.setdefault("tag_pairs", []).append(
                            (source_tag, sink_tag)
                        )
                    else:
                        G.add_edge(
                            src_node.node_id,
                            snk_node.node_id,
                            risk_type=risk_type,
                            risk_types=[risk_type],
                            source_tag=source_tag,
                            sink_tag=sink_tag,
                            edge_id=edge_id,
                            edge_ids=[edge_id],
                            tag_pairs=[(source_tag, sink_tag)],
                        )
                        edge_count += 1

    logger.info(
        "Built DiGraph: %d nodes, %d edges (from %d RULE_SPECS)",
        G.number_of_nodes(),
        edge_count,
        len(RULE_SPECS),
    )
    return G


# ── Path enumeration ───────────────────────────────────────────────────

def enumerate_attack_paths(
    G: nx.DiGraph,
    *,
    max_hops: int = 5,
    max_paths_per_pair: int = 3,
    budget_per_hop: int = 50_000,
    seed: int = 42,
) -> list[list[str]]:
    """Enumerate simple paths from entry → exit using iterative BFS hop extension.

    Unlike ``nx.all_simple_paths`` (which explodes on dense graphs), this
    maintains a bounded frontier at each hop. For each hop h:

      1. Extend every path p in frontier(h-1) by one successor edge.
      2. Emit extended paths whose tail is an exit node (bounded by
         ``max_paths_per_pair`` per (entry, exit) pair and by
         ``budget_per_hop`` globally).
      3. Keep at most ``budget_per_hop`` extended paths as frontier(h)
         for the next hop (sampled randomly to avoid entry-node skew).

    This gives O(budget_per_hop × avg_out_degree) work per hop, making
    the pipeline tractable on the 87M-edge / 47k-node production graph.

    Args:
        G: The skill graph.
        max_hops: Maximum number of edges in a path.
        max_paths_per_pair: Cap paths per (entry, exit) pair per hop length.
        budget_per_hop: Max paths to collect per hop length AND max
            frontier size carried into the next hop.
        seed: RNG seed for frontier shuffling / successor iteration order.

    Returns:
        List of paths, where each path is a list of node_ids.
    """
    rng = random.Random(seed)

    entry_nodes = [
        nid for nid, d in G.nodes(data=True) if d.get("is_entry")
    ]
    exit_set = {
        nid for nid, d in G.nodes(data=True) if d.get("is_exit")
    }

    logger.info(
        "Enumerating paths (iterative BFS): %d entry nodes, %d exit nodes, "
        "max_hops=%d, budget_per_hop=%d",
        len(entry_nodes),
        len(exit_set),
        max_hops,
        budget_per_hop,
    )

    # Frontier starts as length-0 paths (each entry node alone).
    # We shuffle so that when we truncate by budget, we don't bias
    # toward the first few entry nodes.
    rng.shuffle(entry_nodes)
    frontier: list[list[str]] = [[e] for e in entry_nodes]

    all_paths: list[list[str]] = []

    for hop in range(1, max_hops + 1):
        new_frontier: list[list[str]] = []
        hop_paths: list[list[str]] = []
        pair_counts: Counter[tuple[str, str]] = Counter()
        frontier_full = False
        emit_full = False

        # Shuffle frontier so the entries/paths explored first are random.
        rng.shuffle(frontier)

        for path in frontier:
            if emit_full and frontier_full:
                break
            last = path[-1]
            visited = set(path)
            # Materialize successors so we can shuffle; bounded by
            # node out-degree (avg ~1800 on the production graph).
            succs = list(G.successors(last))
            rng.shuffle(succs)

            for succ in succs:
                if succ in visited:
                    continue
                new_path = path + [succ]

                # Collect as a hop-h result if tail is an exit node.
                if not emit_full and succ in exit_set:
                    pair_key = (new_path[0], succ)
                    if pair_counts[pair_key] < max_paths_per_pair:
                        hop_paths.append(new_path)
                        pair_counts[pair_key] += 1
                        if len(hop_paths) >= budget_per_hop:
                            emit_full = True

                # Keep for next hop's frontier.
                if hop < max_hops and not frontier_full:
                    new_frontier.append(new_path)
                    if len(new_frontier) >= budget_per_hop:
                        frontier_full = True

                if emit_full and frontier_full:
                    break

        logger.info(
            "Hop %d: %d paths emitted, next-frontier size=%d (entry-pairs=%d)",
            hop,
            len(hop_paths),
            len(new_frontier),
            len(pair_counts),
        )
        all_paths.extend(hop_paths)

        if hop >= max_hops or not new_frontier:
            break
        frontier = new_frontier

    hop_dist = Counter(len(p) - 1 for p in all_paths)
    logger.info(
        "Enumerated %d paths total. Hop distribution: %s",
        len(all_paths),
        dict(sorted(hop_dist.items())),
    )
    return all_paths


# ── Pruning ────────────────────────────────────────────────────────────

def _compute_path_embedding_score(
    path: list[str],
    embeddings: dict[str, list[float]],
    node_to_skill: dict[str, str],
) -> float:
    """Average pairwise cosine similarity along the path.

    Higher score = more thematically coherent chain.
    Returns 0.0 if embeddings are unavailable.
    """
    skill_ids = [node_to_skill.get(nid, "") for nid in path]
    vecs = []
    for sid in skill_ids:
        emb = embeddings.get(sid)
        if emb:
            vecs.append(emb)
    if len(vecs) < 2:
        return 0.0

    mat = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat = mat / norms

    n = len(mat)
    total = 0.0
    pairs = 0
    for i in range(n - 1):
        sim = float(mat[i] @ mat[i + 1])
        total += sim
        pairs += 1
    return total / pairs if pairs else 0.0


def prune_paths(
    paths: list[list[str]],
    G: nx.DiGraph,
    *,
    embeddings: dict[str, list[float]] | None = None,
    min_embedding_score: float = 0.15,
    max_per_hop_stratum: int = 5000,
    max_per_cluster_signature: int = 5,
    seed: int = 42,
) -> list[list[str]]:
    """Prune enumerated paths by embedding coherence, cluster diversity, sampling.

    Steps:
      1. Filter paths below embedding similarity threshold
      2. Deduplicate by cluster signature (keep max_per_cluster_signature per sig)
      3. Sample down to max_per_hop_stratum per hop length
    """
    node_to_skill = {
        nid: d["skill_id"] for nid, d in G.nodes(data=True)
    }

    # Step 1: Embedding coherence filter
    if embeddings:
        scored: list[tuple[list[str], float]] = []
        for path in paths:
            score = _compute_path_embedding_score(path, embeddings, node_to_skill)
            if score >= min_embedding_score or len(path) == 2:
                # Always keep 1-hop paths (direct edges)
                scored.append((path, score))
        # Sort by score descending within same hop length
        scored.sort(key=lambda x: (-len(x[0]), -x[1]))
        paths = [p for p, _ in scored]
        logger.info("After embedding filter: %d paths", len(paths))

    # Step 2: Cluster signature dedup
    node_to_cluster = {
        nid: d["cluster_id"] for nid, d in G.nodes(data=True)
    }
    cluster_sig_counts: Counter[str] = Counter()
    deduped: list[list[str]] = []
    for path in paths:
        sig = "|".join(node_to_cluster.get(nid, "?") for nid in path)
        cluster_sig_counts[sig] += 1
        if cluster_sig_counts[sig] <= max_per_cluster_signature:
            deduped.append(path)
    paths = deduped
    logger.info("After cluster dedup: %d paths", len(paths))

    # Step 3: Per-hop-length sampling
    rng = random.Random(seed)
    by_hop: dict[int, list[list[str]]] = {}
    for path in paths:
        hop = len(path) - 1
        by_hop.setdefault(hop, []).append(path)

    sampled: list[list[str]] = []
    for hop in sorted(by_hop.keys()):
        group = by_hop[hop]
        if len(group) <= max_per_hop_stratum:
            sampled.extend(group)
        else:
            rng.shuffle(group)
            sampled.extend(group[:max_per_hop_stratum])

    logger.info("After per-hop sampling: %d paths", len(sampled))
    return sampled


# ── Path → SubgraphManifestRecord conversion ──────────────────────────

def _determine_attack_type(path: list[str], G: nx.DiGraph) -> str:
    """Determine primary attack type from the last edge's risk_type."""
    if len(path) < 2:
        return "unknown"
    last_src, last_dst = path[-2], path[-1]
    edata = G.get_edge_data(last_src, last_dst)
    if edata:
        return str(edata.get("risk_type", "unknown"))
    return "unknown"


def _path_node_role(idx: int, path_len: int) -> str:
    """Assign a role label to a node based on its position in the path."""
    if idx == 0:
        return "source"
    if idx == path_len - 1:
        return "sink"
    return "bridge"


def paths_to_manifests(
    paths: list[list[str]],
    G: nx.DiGraph,
) -> list[SubgraphManifestRecord]:
    """Convert enumerated paths to SubgraphManifestRecord objects."""
    manifests: list[SubgraphManifestRecord] = []

    for path in paths:
        hop_length = len(path) - 1
        attack_type = _determine_attack_type(path, G)

        # Determine subgraph type based on hop count
        if hop_length == 1:
            sg_type = "direct"
        elif hop_length == 2:
            sg_type = "chain_2hop"
        else:
            sg_type = f"chain_{hop_length}hop"

        # Build node refs
        node_refs = [
            SubgraphNodeRef(
                node_id=nid,
                role=_path_node_role(i, len(path)),
            )
            for i, nid in enumerate(path)
        ]

        # Build edge refs
        edge_refs = []
        edge_ids = []
        for i in range(len(path) - 1):
            edata = G.get_edge_data(path[i], path[i + 1]) or {}
            eid = str(edata.get("edge_id", f"edge:{path[i]}->{path[i+1]}"))
            edge_refs.append(SubgraphEdgeRef(edge_id=eid, order_index=i))
            edge_ids.append(eid)

        # Collect metadata
        cluster_ids = [
            G.nodes[nid].get("cluster_id", "") for nid in path
        ]
        heat_sig = [
            G.nodes[nid].get("heat_tier", "unknown") for nid in path
        ]
        skill_ids = [
            G.nodes[nid].get("skill_id", "") for nid in path
        ]

        # Subgraph ID encodes the skill chain for uniqueness
        sg_id = f"subgraph:{sg_type}:{'>'.join(skill_ids)}"

        manifests.append(
            SubgraphManifestRecord(
                subgraph_id=sg_id,
                schema_version=SCHEMA_VERSION,
                subgraph_type=sg_type,
                primary_attack_type=attack_type,
                hop_length=hop_length,
                nodes=node_refs,
                edges=edge_refs,
                role_signature=[ref.role for ref in node_refs],
                heat_signature=heat_sig,
                difficulty_signature=(
                    ["explicit"] if hop_length <= 2 else ["multi_hop"]
                ),
                cluster_ids=cluster_ids,
                selection_reason="graph_path_enumeration",
                lineage={
                    "edge_ids": edge_ids,
                    "skill_chain": skill_ids,
                    "hop_length": hop_length,
                },
            )
        )

    return manifests


# ── Evaluation selection ───────────────────────────────────────────────

def select_evaluation_subgraphs(
    candidates: list[SubgraphManifestRecord],
    *,
    max_per_stratum: int = 10,
) -> list[SubgraphManifestRecord]:
    """Coverage-guided selection: keep max_per_stratum per (hop_length, attack_type).

    Within each stratum, prefer cluster diversity.
    """
    strata: dict[tuple[int, str], list[SubgraphManifestRecord]] = {}
    for sg in candidates:
        key = (sg.hop_length, sg.primary_attack_type)
        strata.setdefault(key, []).append(sg)

    selected: list[SubgraphManifestRecord] = []
    for _, group in sorted(strata.items()):
        if len(group) <= max_per_stratum:
            selected.extend(group)
            continue
        # Prefer cluster diversity
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
            budget = max_per_stratum - len(diverse)
            selected.extend(remainder[:budget])

    return selected


# ── Main pipeline entry point ──────────────────────────────────────────

def build_graph_dataset(
    records: list[SkillFingerprintRecord],
    output_dir: Path,
    *,
    max_hops: int = 5,
    max_paths_per_pair: int = 3,
    budget_per_hop: int = 50_000,
    min_embedding_score: float = 0.15,
    max_per_hop_stratum: int = 5000,
    max_per_cluster_signature: int = 5,
    max_per_eval_stratum: int = 10,
    seed: int = 42,
    theme_threshold: float | None = None,
) -> dict[str, Path]:
    """Full pipeline: nodes → DiGraph → paths → prune → manifests → splits.

    This replaces ``build_subgraph_dataset`` with a graph-based approach
    that supports arbitrary hop lengths.
    """
    output_dir = ensure_directory(output_dir)

    # Step 1: Materialize nodes
    nodes = materialize_nodes(records)
    logger.info("Materialized %d nodes", len(nodes))

    # Step 2: Compute theme similarity pairs (optional)
    embeddings = {
        rec.skill_id: rec.doc_embedding
        for rec in records
        if rec.doc_embedding
    }
    theme_pairs: set[tuple[str, str]] | None = None
    if embeddings:
        from .subgraph_pipeline import _build_similarity_lookup, THEME_SIMILARITY_THRESHOLD
        thr = theme_threshold if theme_threshold is not None else THEME_SIMILARITY_THRESHOLD
        theme_pairs = _build_similarity_lookup(embeddings, thr)
        logger.info(
            "Computed %d theme-similar pairs (threshold=%.2f)",
            len(theme_pairs) if theme_pairs else 0,
            thr,
        )

    # Step 3: Build directed graph
    G = build_digraph(nodes, theme_pairs=theme_pairs)

    # Step 3b: Checkpoint — serialize nodes and edges immediately so the
    # ~30-minute graph build isn't lost if path enumeration fails.
    nodes_by_id = {n.node_id: n for n in nodes}
    edges: list[RiskGraphEdgeRecord] = []
    for u, v, edata in G.edges(data=True):
        src_node = nodes_by_id.get(u)
        snk_node = nodes_by_id.get(v)
        if not src_node or not snk_node:
            continue
        first_tag_pair = edata.get("tag_pairs", [("", "")])[0]
        edges.append(
            RiskGraphEdgeRecord(
                edge_id=str(edata.get("edge_id", f"edge:{u}->{v}")),
                source_node_id=u,
                sink_node_id=v,
                source_skill_id=src_node.skill_id,
                sink_skill_id=snk_node.skill_id,
                source_tag=str(edata.get("source_tag", "")),
                sink_tag=str(edata.get("sink_tag", "")),
                risk_type=str(edata.get("risk_type", "")),
                edge_kind="direct",
                confidence=1.0,
                support_tags=list(first_tag_pair),
                cluster_pair=[src_node.cluster_id, snk_node.cluster_id],
                evidence_stub=f"rule:{edata.get('source_tag', '')}->{edata.get('sink_tag', '')}",
                selection_reason="graph_edge",
            )
        )

    nodes_path = write_jsonl_records(
        output_dir / "nodes.jsonl",
        [node.to_dict() for node in nodes],
    )
    edges_path = write_jsonl_records(
        output_dir / "edges.jsonl",
        [edge.to_dict() for edge in edges],
    )
    logger.info(
        "Checkpoint: wrote %d nodes, %d edges before path enumeration",
        len(nodes),
        len(edges),
    )

    # Step 4: Enumerate attack paths (iterative BFS, bounded frontier)
    all_paths = enumerate_attack_paths(
        G,
        max_hops=max_hops,
        max_paths_per_pair=max_paths_per_pair,
        budget_per_hop=budget_per_hop,
        seed=seed,
    )

    # Step 5: Prune
    pruned_paths = prune_paths(
        all_paths,
        G,
        embeddings=embeddings if embeddings else None,
        min_embedding_score=min_embedding_score,
        max_per_hop_stratum=max_per_hop_stratum,
        max_per_cluster_signature=max_per_cluster_signature,
        seed=seed,
    )

    # Step 6: Convert to manifests
    subgraphs = paths_to_manifests(pruned_paths, G)
    logger.info("Generated %d subgraph manifests", len(subgraphs))

    # Step 8: Select eval subgraphs and split
    selected = select_evaluation_subgraphs(subgraphs, max_per_stratum=max_per_eval_stratum)
    splits = split_subgraphs(selected, seed=seed)

    # Step 9: Compute stats
    stats = build_subgraph_stats(nodes, edges, subgraphs)

    # Step 10: Write remaining outputs (nodes/edges already written above)
    manifest_path = write_jsonl_records(
        output_dir / "subgraph_manifest.jsonl",
        [sg.to_dict() for sg in subgraphs],
    )
    stats_path = write_json(output_dir / "subgraph_stats.json", stats.to_dict())
    coverage_path = write_markdown(
        output_dir / "subgraph_coverage.md",
        render_subgraph_coverage(stats),
    )

    # Graph metadata
    graph_meta = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "entry_nodes": sum(1 for _, d in G.nodes(data=True) if d.get("is_entry")),
        "exit_nodes": sum(1 for _, d in G.nodes(data=True) if d.get("is_exit")),
        "total_paths_enumerated": len(all_paths),
        "total_paths_after_pruning": len(pruned_paths),
        "total_subgraphs": len(subgraphs),
        "max_hops": max_hops,
        "hop_distribution": dict(Counter(sg.hop_length for sg in subgraphs)),
        "attack_type_distribution": dict(Counter(sg.primary_attack_type for sg in subgraphs)),
    }
    graph_meta_path = write_json(output_dir / "graph_meta.json", graph_meta)

    # Splits
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
        "subgraph_stats": stats_path,
        "subgraph_coverage": coverage_path,
        "graph_meta": graph_meta_path,
        "splits_train": splits_train,
        "splits_dev": splits_dev,
        "splits_eval": splits_eval,
    }
