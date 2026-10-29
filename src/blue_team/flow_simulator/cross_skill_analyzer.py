"""SkillProbe P3 — Cross-skill Flow Simulator.

Analyzes data flow across multiple skills in a composition subgraph to detect
attack chains that single-skill audit cannot identify.

Architecture:
  1. Parse subgraph edges to build a cross-skill data flow graph
  2. For subgraphs where individual audits missed some skills, use LLM to
     assess whether the combined data flow constitutes an attack chain
  3. Return per-skill verdict overrides for individually-benign skills that
     participate in a verified attack chain
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from core.models import SkillArtifact
from shared.loaders.skill_loader import load_skill_artifact

from ..llm_client import chat_json, is_available

logger = logging.getLogger(__name__)

_FLOW_SIM_PROMPT = """\
You are a security auditor analyzing a CROSS-SKILL ATTACK CHAIN.

Individual skill audits found some skills in this chain to look benign on their own.
However, when composed together, they may form a dangerous data flow path.

## Subgraph Attack Type: {attack_type}

## Data Flow Edges (how data moves between skills):
{edges_description}

## Skills in the Chain:
{skills_description}

## Task:
Analyze whether these skills, when composed together, form a real attack chain.

Key questions:
1. Can data actually flow from the source skill's output to the sink skill's input?
2. Does the combined flow create a capability that no single skill has alone?
3. Is there a plausible attack scenario where an adversary exploits this composition?

Common attack patterns:
- Skill A reads credentials → Skill B sends them to attacker server (data exfiltration)
- Skill A fetches untrusted content → Skill B executes it as code (remote code execution)
- Skill A modifies local files → Skill B reads those files for persistent influence (persistence)
- Skill A collects user data → Skill B stores it in attacker-controlled DB (data poisoning)

Answer strictly in JSON:
{{
  "is_attack_chain": true/false,
  "confidence": "high"/"medium"/"low",
  "attack_narrative": "one paragraph describing the attack flow if true, or why it's benign if false",
  "risky_skill_ids": ["skill_id_1", "skill_id_2"],
  "reasoning": "brief technical reasoning"
}}"""


@dataclass(slots=True)
class FlowSimResult:
    """Result of cross-skill flow simulation for one subgraph."""

    subgraph_id: str
    is_attack_chain: bool = False
    confidence: str = "low"
    attack_narrative: str = ""
    risky_skill_ids: list[str] = field(default_factory=list)
    reasoning: str = ""
    used_llm: bool = False
    error: str = ""


@dataclass(slots=True)
class SubgraphContext:
    """Parsed subgraph with edge and skill information."""

    subgraph_id: str
    attack_type: str
    node_skill_ids: list[str]
    edges: list[dict[str, str]]
    lineage: dict[str, object]


def _parse_edge_description(edges: list[dict[str, str]]) -> str:
    """Convert subgraph edges to human-readable description."""
    lines = []
    for edge in edges:
        edge_id = edge.get("edge_id", "")
        # edge format: edge:skillA:source_tag->skillB:sink_tag
        cleaned = edge_id.removeprefix("edge:")
        if "->" in cleaned:
            source_part, sink_part = cleaned.split("->", 1)
            lines.append(f"  {source_part} → {sink_part}")
        else:
            lines.append(f"  {cleaned}")
    return "\n".join(lines) if lines else "  (no edges)"


def _build_skill_summary(skill: SkillArtifact) -> str:
    """Build a compact summary of a skill for the LLM prompt."""
    doc_preview = skill.skill_doc[:1500]
    code_preview = ""
    budget = 2000
    for path, content in skill.code_files.items():
        if budget <= 0:
            break
        snippet = content[:budget]
        code_preview += f"\n### {path}\n```\n{snippet}\n```"
        budget -= len(snippet)

    return (
        f"### {skill.skill_id}\n"
        f"Source tags: {skill.source_tags}\n"
        f"Sink tags: {skill.sink_tags}\n"
        f"SKILL.md preview:\n{doc_preview}\n"
        f"{code_preview}"
    )


class CrossSkillFlowSimulator:
    """Analyze cross-skill data flow to detect composition attacks."""

    def __init__(self, skills_root: str | Path) -> None:
        self._skills_root = Path(skills_root)

    def simulate(
        self,
        subgraph: SubgraphContext,
        *,
        layer: str = "flow_sim",
    ) -> FlowSimResult:
        """Run flow simulation on a single subgraph."""
        if not is_available(layer):
            return FlowSimResult(
                subgraph_id=subgraph.subgraph_id,
                error="LLM not available for flow simulation",
            )

        # Load skills
        skills: list[SkillArtifact] = []
        for sid in subgraph.node_skill_ids:
            skill_dir = self._skills_root / sid
            if not skill_dir.exists():
                continue
            skill = load_skill_artifact(skill_dir)
            if skill is not None:
                skills.append(skill)

        if len(skills) < 2:
            return FlowSimResult(
                subgraph_id=subgraph.subgraph_id,
                error=f"Need >=2 skills for cross-skill analysis, got {len(skills)}",
            )

        edges_desc = _parse_edge_description(subgraph.edges)
        skills_desc = "\n\n".join(_build_skill_summary(s) for s in skills)

        prompt = _FLOW_SIM_PROMPT.format(
            attack_type=subgraph.attack_type,
            edges_description=edges_desc,
            skills_description=skills_desc,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            result = chat_json(messages, layer=layer, temperature=0.1, timeout=120.0)
        except Exception as exc:
            logger.warning(
                "Flow simulation failed for %s: %s",
                subgraph.subgraph_id, exc,
            )
            return FlowSimResult(
                subgraph_id=subgraph.subgraph_id,
                error=str(exc),
            )

        raw_risky = result.get("risky_skill_ids", [])
        risky_ids = [str(s) for s in raw_risky] if isinstance(raw_risky, list) else []

        return FlowSimResult(
            subgraph_id=subgraph.subgraph_id,
            is_attack_chain=bool(result.get("is_attack_chain", False)),
            confidence=str(result.get("confidence", "low")),
            attack_narrative=str(result.get("attack_narrative", "")),
            risky_skill_ids=risky_ids,
            reasoning=str(result.get("reasoning", "")),
            used_llm=True,
        )

    def simulate_batch(
        self,
        subgraphs: list[SubgraphContext],
        *,
        layer: str = "flow_sim",
    ) -> dict[str, FlowSimResult]:
        """Run flow simulation on multiple subgraphs."""
        return {
            sg.subgraph_id: self.simulate(sg, layer=layer)
            for sg in subgraphs
        }
