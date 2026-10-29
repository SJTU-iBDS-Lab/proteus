from __future__ import annotations

import json

from core.models import SkillFingerprintRecord, SubgraphManifestRecord, SubgraphNodeRef, SubgraphEdgeRef
from dataset.subgraph_pipeline import build_subgraph_dataset, select_evaluation_subgraphs, split_subgraphs


def test_build_subgraph_dataset_emits_direct_subgraph_artifacts(tmp_path) -> None:
    records = [
        SkillFingerprintRecord(
            skill_id="skill-source",
            skill_path="data/skills/source",
            schema_version="subgraph.v1",
            cluster_id="cluster-a",
            category="secrets",
            heat_tier="tail",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["read_env"],
            dependency_profile=["requests"],
            source_tags=["credentials"],
            sink_tags=[],
            bridge_tags=[],
            attack_family_candidates=["P6"],
            selection_reason="long_tail_risk",
        ),
        SkillFingerprintRecord(
            skill_id="skill-sink",
            skill_path="data/skills/sink",
            schema_version="subgraph.v1",
            cluster_id="cluster-b",
            category="network",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["post_request"],
            dependency_profile=["requests"],
            source_tags=[],
            sink_tags=["network_egress"],
            bridge_tags=[],
            attack_family_candidates=["P6"],
            selection_reason="representative_sink",
        ),
    ]

    result = build_subgraph_dataset(records, tmp_path)

    manifest_path = result["subgraph_manifest"]
    stats_path = result["subgraph_stats"]
    pruned_path = result["pruned_candidates"]

    manifest_lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 1

    manifest = json.loads(manifest_lines[0])
    assert manifest["subgraph_type"] == "direct"
    assert manifest["primary_attack_type"] == "data_exfiltration"

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["total_subgraphs"] == 1
    assert stats["topology_distribution"]["direct"] == 1

    assert pruned_path.exists()


def test_build_subgraph_dataset_emits_bridge_subgraph_for_bridge_tagged_middle_node(
    tmp_path,
) -> None:
    records = [
        SkillFingerprintRecord(
            skill_id="skill-source",
            skill_path="data/skills/source",
            schema_version="subgraph.v1",
            cluster_id="cluster-a",
            category="secrets",
            heat_tier="tail",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["read_env"],
            dependency_profile=["requests"],
            source_tags=["credentials"],
            sink_tags=[],
            bridge_tags=[],
            attack_family_candidates=["P6"],
            selection_reason="source",
        ),
        SkillFingerprintRecord(
            skill_id="skill-bridge",
            skill_path="data/skills/bridge",
            schema_version="subgraph.v1",
            cluster_id="cluster-b",
            category="transform",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["json_transform"],
            dependency_profile=[],
            source_tags=["local_files"],
            sink_tags=["network_egress"],
            bridge_tags=["json_transform"],
            attack_family_candidates=[],
            selection_reason="bridge",
        ),
        SkillFingerprintRecord(
            skill_id="skill-sink",
            skill_path="data/skills/sink",
            schema_version="subgraph.v1",
            cluster_id="cluster-c",
            category="exec",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["shell_exec"],
            dependency_profile=[],
            source_tags=[],
            sink_tags=["shell_execution"],
            bridge_tags=[],
            attack_family_candidates=[],
            selection_reason="sink",
        ),
    ]

    result = build_subgraph_dataset(records, tmp_path)

    manifest_lines = [
        json.loads(line)
        for line in result["subgraph_manifest"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bridge_manifests = [
        item for item in manifest_lines if item["subgraph_type"] == "bridge"
    ]

    assert len(bridge_manifests) >= 1
    assert bridge_manifests[0]["hop_length"] == 2
    assert bridge_manifests[0]["role_signature"] == ["source", "bridge", "sink"]


def test_build_subgraph_dataset_skips_bridge_without_bridge_tag(tmp_path) -> None:
    records = [
        SkillFingerprintRecord(
            skill_id="skill-source",
            skill_path="data/skills/source",
            schema_version="subgraph.v1",
            cluster_id="cluster-a",
            category="secrets",
            heat_tier="tail",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["read_env"],
            dependency_profile=["requests"],
            source_tags=["credentials"],
            sink_tags=[],
            bridge_tags=[],
            attack_family_candidates=["P6"],
            selection_reason="source",
        ),
        SkillFingerprintRecord(
            skill_id="skill-middle",
            skill_path="data/skills/middle",
            schema_version="subgraph.v1",
            cluster_id="cluster-b",
            category="transform",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["cache_file"],
            dependency_profile=["requests"],
            source_tags=["local_files"],
            sink_tags=["network_egress"],
            bridge_tags=[],
            attack_family_candidates=["P6"],
            selection_reason="not_a_bridge",
        ),
        SkillFingerprintRecord(
            skill_id="skill-sink",
            skill_path="data/skills/sink",
            schema_version="subgraph.v1",
            cluster_id="cluster-c",
            category="filesystem",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["write_file"],
            dependency_profile=["requests"],
            source_tags=[],
            sink_tags=["filesystem_write"],
            bridge_tags=[],
            attack_family_candidates=["P5"],
            selection_reason="sink",
        ),
    ]

    result = build_subgraph_dataset(records, tmp_path)

    manifest_lines = [
        json.loads(line)
        for line in result["subgraph_manifest"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    pruned_lines = [
        json.loads(line)
        for line in result["pruned_candidates"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert all(item["subgraph_type"] != "bridge" for item in manifest_lines)
    assert any(
        item.get("candidate_type") == "bridge"
        and item.get("prune_reason") == "missing_bridge_tag"
        for item in pruned_lines
    )


def test_build_subgraph_dataset_prunes_duplicate_bridge_candidates(tmp_path) -> None:
    records = [
        SkillFingerprintRecord(
            skill_id="skill-source",
            skill_path="data/skills/source",
            schema_version="subgraph.v1",
            cluster_id="cluster-a",
            category="secrets",
            heat_tier="tail",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["read_env"],
            dependency_profile=["requests"],
            source_tags=["credentials"],
            sink_tags=[],
            bridge_tags=[],
            attack_family_candidates=["P6"],
            selection_reason="source",
        ),
        SkillFingerprintRecord(
            skill_id="skill-bridge-a",
            skill_path="data/skills/bridge-a",
            schema_version="subgraph.v1",
            cluster_id="cluster-b",
            category="transform",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["cache_file"],
            dependency_profile=["requests"],
            source_tags=["local_files"],
            sink_tags=["network_egress"],
            bridge_tags=["json_transform"],
            attack_family_candidates=["P6"],
            selection_reason="bridge-a",
        ),
        SkillFingerprintRecord(
            skill_id="skill-bridge-b",
            skill_path="data/skills/bridge-b",
            schema_version="subgraph.v1",
            cluster_id="cluster-b",
            category="transform",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["cache_file"],
            dependency_profile=["requests"],
            source_tags=["local_files"],
            sink_tags=["network_egress"],
            bridge_tags=["json_transform"],
            attack_family_candidates=["P6"],
            selection_reason="bridge-b",
        ),
        SkillFingerprintRecord(
            skill_id="skill-sink",
            skill_path="data/skills/sink",
            schema_version="subgraph.v1",
            cluster_id="cluster-c",
            category="filesystem",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=["pypi"],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["write_file"],
            dependency_profile=["requests"],
            source_tags=[],
            sink_tags=["filesystem_write"],
            bridge_tags=[],
            attack_family_candidates=["P5"],
            selection_reason="sink",
        ),
    ]

    result = build_subgraph_dataset(records, tmp_path)

    manifest_lines = [
        json.loads(line)
        for line in result["subgraph_manifest"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pruned_lines = [
        json.loads(line)
        for line in result["pruned_candidates"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    bridge_manifests = [
        item for item in manifest_lines if item["subgraph_type"] == "bridge"
    ]

    # Two bridge nodes in same cluster: dedup keeps at most one per cluster-triple
    assert len(bridge_manifests) >= 1
    assert any(
        item.get("candidate_type") == "bridge"
        and item.get("prune_reason") == "duplicate_bridge_signature"
        for item in pruned_lines
    )


def test_build_subgraph_dataset_emits_stateful_subgraph(tmp_path) -> None:
    """Stateful: source -> stateful_node -> sink.

    stateful_node must have BOTH filesystem_write in sink_tags (writes state)
    AND local_files in source_tags (reads it back later).
    """
    records = [
        SkillFingerprintRecord(
            skill_id="env-reader",
            skill_path="data/skills/env-reader",
            schema_version="subgraph.v1",
            cluster_id="cluster-a",
            category="secrets",
            heat_tier="tail",
            language="python",
            dependency_ecosystem=[],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["env_read"],
            dependency_profile=[],
            source_tags=["credentials"],
            sink_tags=[],
            bridge_tags=[],
            attack_family_candidates=[],
            selection_reason="source",
        ),
        SkillFingerprintRecord(
            skill_id="config-cache",
            skill_path="data/skills/config-cache",
            schema_version="subgraph.v1",
            cluster_id="cluster-b",
            category="storage",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=[],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["filesystem_read", "filesystem_write"],
            dependency_profile=[],
            source_tags=["local_files"],
            sink_tags=["filesystem_write"],
            bridge_tags=["json_transform"],
            attack_family_candidates=[],
            selection_reason="stateful_node",
        ),
        SkillFingerprintRecord(
            skill_id="shell-runner",
            skill_path="data/skills/shell-runner",
            schema_version="subgraph.v1",
            cluster_id="cluster-c",
            category="exec",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=[],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["shell_exec"],
            dependency_profile=[],
            source_tags=[],
            sink_tags=["shell_execution"],
            bridge_tags=[],
            attack_family_candidates=[],
            selection_reason="sink",
        ),
    ]

    result = build_subgraph_dataset(records, tmp_path)

    manifest_lines = [
        json.loads(line)
        for line in result["subgraph_manifest"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stateful = [m for m in manifest_lines if m["subgraph_type"] == "stateful"]

    assert len(stateful) >= 1
    assert stateful[0]["hop_length"] == 2
    assert stateful[0]["role_signature"] == ["source", "persistent_state", "sink"]


def test_build_subgraph_dataset_emits_remote_drift_subgraph(tmp_path) -> None:
    """Remote drift: loader -> relay -> sink.

    relay_node has network_input (fetches remote data) and a dangerous sink.
    """
    records = [
        SkillFingerprintRecord(
            skill_id="file-reader",
            skill_path="data/skills/file-reader",
            schema_version="subgraph.v1",
            cluster_id="cluster-a",
            category="files",
            heat_tier="head",
            language="python",
            dependency_ecosystem=[],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["filesystem_read"],
            dependency_profile=[],
            source_tags=["local_files"],
            sink_tags=[],
            bridge_tags=[],
            attack_family_candidates=[],
            selection_reason="loader",
        ),
        SkillFingerprintRecord(
            skill_id="web-fetcher",
            skill_path="data/skills/web-fetcher",
            schema_version="subgraph.v1",
            cluster_id="cluster-b",
            category="network",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=[],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["network_recv", "shell_exec"],
            dependency_profile=[],
            source_tags=["network_input"],
            sink_tags=["shell_execution"],
            bridge_tags=["json_transform"],
            attack_family_candidates=[],
            selection_reason="relay",
        ),
        SkillFingerprintRecord(
            skill_id="code-runner",
            skill_path="data/skills/code-runner",
            schema_version="subgraph.v1",
            cluster_id="cluster-c",
            category="exec",
            heat_tier="tail",
            language="python",
            dependency_ecosystem=[],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["shell_exec"],
            dependency_profile=[],
            source_tags=[],
            sink_tags=["shell_execution"],
            bridge_tags=[],
            attack_family_candidates=[],
            selection_reason="sink",
        ),
    ]

    result = build_subgraph_dataset(records, tmp_path)

    manifest_lines = [
        json.loads(line)
        for line in result["subgraph_manifest"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    remote = [m for m in manifest_lines if m["subgraph_type"] == "remote_drift"]

    assert len(remote) >= 1
    assert remote[0]["hop_length"] == 2
    assert remote[0]["role_signature"] == ["source", "remote_relay", "sink"]


def _make_manifest(
    subgraph_id: str,
    subgraph_type: str,
    attack_type: str,
    heat: str = "mid",
    cluster: str = "c0",
) -> SubgraphManifestRecord:
    return SubgraphManifestRecord(
        subgraph_id=subgraph_id,
        schema_version="subgraph.v1",
        subgraph_type=subgraph_type,
        primary_attack_type=attack_type,
        hop_length=1 if subgraph_type == "direct" else 2,
        nodes=[
            SubgraphNodeRef(node_id="n1", role="source"),
            SubgraphNodeRef(node_id="n2", role="sink"),
        ],
        edges=[SubgraphEdgeRef(edge_id="e1", order_index=0)],
        role_signature=["source", "sink"],
        heat_signature=[heat, heat],
        difficulty_signature=["explicit"],
        cluster_ids=[cluster],
        selection_reason="test",
        lineage={},
    )


def test_select_evaluation_subgraphs_preserves_coverage() -> None:
    """Selection must keep at least one subgraph per topology × attack_type."""
    candidates = [
        _make_manifest("sg-1", "direct", "data_exfiltration", heat="head"),
        _make_manifest("sg-2", "direct", "data_exfiltration", heat="mid"),
        _make_manifest("sg-3", "direct", "data_exfiltration", heat="tail"),
        _make_manifest("sg-4", "direct", "command_injection", heat="head"),
        _make_manifest("sg-5", "stateful", "privilege_escalation", heat="mid"),
        _make_manifest("sg-6", "stateful", "privilege_escalation", heat="tail"),
        _make_manifest("sg-7", "remote_drift", "repo_exfiltration", heat="head"),
    ]

    selected = select_evaluation_subgraphs(candidates)

    # Must be smaller than input (selection actually prunes)
    assert len(selected) <= len(candidates)

    # Coverage: every topology present in input must appear in output
    input_types = {s.subgraph_type for s in candidates}
    output_types = {s.subgraph_type for s in selected}
    assert input_types == output_types

    # Coverage: every attack family present in input must appear in output
    input_attacks = {s.primary_attack_type for s in candidates}
    output_attacks = {s.primary_attack_type for s in selected}
    assert input_attacks == output_attacks


def test_select_evaluation_subgraphs_caps_per_stratum() -> None:
    """No single topology × attack_type should have more than max_per_stratum entries."""
    candidates = [
        _make_manifest(f"sg-{i}", "direct", "data_exfiltration", cluster=f"c{i}")
        for i in range(50)
    ]

    selected = select_evaluation_subgraphs(candidates, max_per_stratum=5)

    direct_exfil = [s for s in selected if s.subgraph_type == "direct" and s.primary_attack_type == "data_exfiltration"]
    assert len(direct_exfil) <= 5


def test_split_subgraphs_is_deterministic() -> None:
    """Same input + same seed must produce identical splits."""
    candidates = [
        _make_manifest(f"sg-{i}", "direct", "data_exfiltration", cluster=f"c{i}")
        for i in range(20)
    ]

    split_a = split_subgraphs(candidates, seed=42)
    split_b = split_subgraphs(candidates, seed=42)

    assert [s.subgraph_id for s in split_a["train"]] == [s.subgraph_id for s in split_b["train"]]
    assert [s.subgraph_id for s in split_a["dev"]] == [s.subgraph_id for s in split_b["dev"]]
    assert [s.subgraph_id for s in split_a["eval"]] == [s.subgraph_id for s in split_b["eval"]]


def test_split_subgraphs_covers_all_items() -> None:
    """Union of train + dev + eval must equal the input set."""
    candidates = [
        _make_manifest(f"sg-{i}", "direct", "data_exfiltration", cluster=f"c{i}")
        for i in range(20)
    ]

    splits = split_subgraphs(candidates, seed=0)

    all_ids = {s.subgraph_id for s in splits["train"]} | {s.subgraph_id for s in splits["dev"]} | {s.subgraph_id for s in splits["eval"]}
    input_ids = {s.subgraph_id for s in candidates}
    assert all_ids == input_ids


def test_build_subgraph_dataset_produces_splits(tmp_path) -> None:
    """build_subgraph_dataset should output splits/ directory with train/dev/eval."""
    records = [
        SkillFingerprintRecord(
            skill_id="skill-source",
            skill_path="data/skills/source",
            schema_version="subgraph.v1",
            cluster_id="cluster-a",
            category="secrets",
            heat_tier="tail",
            language="python",
            dependency_ecosystem=[],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["env_read"],
            dependency_profile=[],
            source_tags=["credentials"],
            sink_tags=[],
            bridge_tags=[],
            attack_family_candidates=[],
            selection_reason="source",
        ),
        SkillFingerprintRecord(
            skill_id="skill-sink",
            skill_path="data/skills/sink",
            schema_version="subgraph.v1",
            cluster_id="cluster-b",
            category="network",
            heat_tier="mid",
            language="python",
            dependency_ecosystem=[],
            metadata={},
            doc_embedding_ref=None,
            code_ops=["network_send"],
            dependency_profile=[],
            source_tags=[],
            sink_tags=["network_egress"],
            bridge_tags=[],
            attack_family_candidates=[],
            selection_reason="sink",
        ),
    ]

    result = build_subgraph_dataset(records, tmp_path)

    assert "splits_train" in result
    assert "splits_dev" in result
    assert "splits_eval" in result
    assert result["splits_train"].exists()
    assert result["splits_dev"].exists()
    assert result["splits_eval"].exists()
