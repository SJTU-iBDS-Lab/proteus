from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from core.models import SkillFingerprintRecord
from shared.loaders import load_skill_artifact

from .fingerprint_pipeline import DocumentEmbedder, build_skill_fingerprint
from .subgraph_io import ensure_directory, write_jsonl_records, write_markdown

CURATED_SKILL_NAMES: list[str] = [
    # Pure source (no sink_tags)
    "giraffe-guard",
    # Pure sinks (no source_tags)
    "linux-omniscient",
    "1688",
    "arxiv-search-collector",
    # Mixed, no bridge_tags
    "openseti-skill",
    "geekbench",
    # Bridge-capable: credentials + local_files
    "auditclaw-grc",
    "crabukit",
    "weatherpanel-note-aipc",
    "1688-product-search",
    "123pan-upload",
    "ebook-to-md",
    "kie-ai-skill",
    "openclaw-user-data-pack",
    # Bridge-capable: credentials only
    "chaoschain",
    "seedream-image-for-openclaw",
    "unifi-site-manager",
    # Bridge-capable: local_files only
    "tasmota-skill",
    "mystool",
    "expense-invoice-ocr",
]


def select_skill_roots(root: Path, limit: int) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if not (path / "SKILL.md").exists():
            continue
        selected.append(path)
        if len(selected) >= limit:
            break
    return selected


def select_curated_skill_roots(root: Path) -> list[Path]:
    roots: list[Path] = []
    for name in CURATED_SKILL_NAMES:
        skill_dir = root / name
        if not skill_dir.is_dir() or not (skill_dir / "SKILL.md").exists():
            raise FileNotFoundError(f"curated skill not found: {skill_dir}")
        roots.append(skill_dir)
    return roots


def _heat_tier(index: int, total: int) -> str:
    if total <= 0:
        return "tail"
    ratio = index / total
    if ratio < 0.2:
        return "head"
    if ratio < 0.6:
        return "mid"
    return "tail"


def build_candidate_skill_records(
    skill_roots: list[Path], *, embedder: DocumentEmbedder
) -> list[SkillFingerprintRecord]:
    total = len(skill_roots)
    records: list[SkillFingerprintRecord] = []
    for index, skill_root in enumerate(skill_roots):
        artifact = load_skill_artifact(skill_root)
        records.append(
            build_skill_fingerprint(
                artifact,
                embedder=embedder,
                cluster_id=f"cluster-{index:03d}",
                category="demo_subset",
                heat_tier=_heat_tier(index, total),
            )
        )
    return records


def render_mermaid_risk_graph(
    *, nodes: list[dict[str, object]], edges: list[dict[str, object]]
) -> str:
    node_lines: list[str] = []
    edge_lines: list[str] = []
    seen_nodes: set[str] = set()

    for node in nodes:
        node_id = str(node["node_id"])
        skill_id = str(node.get("skill_id", node_id))
        mermaid_id = node_id.replace(":", "_").replace("-", "_")
        if mermaid_id in seen_nodes:
            continue
        seen_nodes.add(mermaid_id)
        node_lines.append(f'{mermaid_id}["{skill_id}"]')

    for edge in edges:
        source = str(edge["source_node_id"]).replace(":", "_").replace("-", "_")
        sink = str(edge["sink_node_id"]).replace(":", "_").replace("-", "_")
        label = f"{edge.get('edge_kind', 'direct')}:{edge.get('risk_type', 'unknown')}"
        edge_lines.append(f"{source} -->|{label}| {sink}")

    lines = ["```mermaid", "flowchart LR", *node_lines, *edge_lines, "```"]
    return "\n".join(lines) + "\n"


def write_candidate_skill_records(
    records: list[SkillFingerprintRecord], output_path: Path
) -> Path:
    return write_jsonl_records(output_path, [record.to_dict() for record in records])


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_mermaid_preview(
    output_dir: Path, *, nodes: list[dict[str, object]], edges: list[dict[str, object]]
) -> Path:
    ensure_directory(output_dir)
    content = "# Subset Risk Graph Preview\n\n" + render_mermaid_risk_graph(
        nodes=nodes, edges=edges
    )
    return write_markdown(output_dir / "subset_graph_preview.md", content)


def _classify_role(
    source_tags: list[str], sink_tags: list[str], bridge_tags: list[str]
) -> str:
    has_src = bool(source_tags)
    has_sink = bool(sink_tags)
    has_bridge = bool(bridge_tags)
    if has_src and not has_sink:
        return "pure-source"
    if has_sink and not has_src:
        return "pure-sink"
    if has_src and has_sink and has_bridge:
        return "bridge-capable"
    if has_src and has_sink:
        return "mixed"
    return "inert"


def render_coverage_report(
    *,
    records: list[SkillFingerprintRecord],
    edges: list[dict[str, object]],
    subgraphs: list[dict[str, object]],
) -> str:
    lines: list[str] = ["# Coverage Report", ""]

    # --- Section 1: Per-skill tag profile ---
    lines.append("## Skill Tag Profiles")
    lines.append("")
    lines.append("| # | Skill | Source | Sink | Bridge | Role |")
    lines.append("|---|-------|--------|------|--------|------|")
    for idx, rec in enumerate(records, 1):
        src = ", ".join(rec.source_tags) or "-"
        sink = ", ".join(rec.sink_tags) or "-"
        bridge = ", ".join(rec.bridge_tags) or "-"
        role = _classify_role(rec.source_tags, rec.sink_tags, rec.bridge_tags)
        lines.append(f"| {idx} | {rec.skill_id} | {src} | {sink} | {bridge} | {role} |")
    lines.append("")

    # --- Section 2: Edge statistics ---
    edge_type_counts: Counter[str] = Counter()
    for edge in edges:
        risk_type = str(edge.get("risk_type", "unknown"))
        edge_type_counts[risk_type] += 1

    lines.append("## Edge Statistics")
    lines.append("")
    lines.append(f"- Total edges: {len(edges)}")
    lines.append("")
    lines.append("| Attack Family | Edge Count |")
    lines.append("|---------------|------------|")
    for family, count in sorted(edge_type_counts.items()):
        lines.append(f"| {family} | {count} |")
    lines.append("")

    # --- Section 3: Subgraph distribution ---
    type_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()
    for sg in subgraphs:
        type_counts[str(sg.get("subgraph_type", "unknown"))] += 1
        attack_counts[str(sg.get("primary_attack_type", "unknown"))] += 1

    lines.append("## Subgraph Distribution")
    lines.append("")
    lines.append(f"- Total subgraphs: {len(subgraphs)}")
    lines.append("")
    lines.append("| Topology | Count |")
    lines.append("|----------|-------|")
    for topo, count in sorted(type_counts.items()):
        lines.append(f"| {topo} | {count} |")
    lines.append("")
    lines.append("| Attack Family | Subgraph Count |")
    lines.append("|---------------|----------------|")
    for family, count in sorted(attack_counts.items()):
        lines.append(f"| {family} | {count} |")
    lines.append("")

    # --- Section 4: Node participation matrix ---
    source_participation: Counter[str] = Counter()
    sink_participation: Counter[str] = Counter()
    bridge_participation: Counter[str] = Counter()

    for sg in subgraphs:
        sg_nodes = sg.get("nodes")
        if not isinstance(sg_nodes, list):
            continue
        for node_ref in sg_nodes:
            if not isinstance(node_ref, dict):
                continue
            node_id = str(node_ref.get("node_id", ""))
            role = str(node_ref.get("role", ""))
            skill_id = node_id.removeprefix("node:")
            if role == "source":
                source_participation[skill_id] += 1
            elif role == "sink":
                sink_participation[skill_id] += 1
            elif role == "bridge":
                bridge_participation[skill_id] += 1

    all_skills = sorted(
        {rec.skill_id for rec in records},
        key=lambda s: (
            -(source_participation.get(s, 0)
              + sink_participation.get(s, 0)
              + bridge_participation.get(s, 0)),
            s,
        ),
    )

    lines.append("## Node Participation in Subgraphs")
    lines.append("")
    lines.append("| Skill | As Source | As Sink | As Bridge | Total |")
    lines.append("|-------|----------|---------|-----------|-------|")
    for skill in all_skills:
        s = source_participation.get(skill, 0)
        k = sink_participation.get(skill, 0)
        b = bridge_participation.get(skill, 0)
        lines.append(f"| {skill} | {s} | {k} | {b} | {s + k + b} |")
    lines.append("")

    return "\n".join(lines)
