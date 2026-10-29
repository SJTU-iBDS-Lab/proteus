"""Chain-prior ranking for ``list_skills`` based on the conspiracy graph.

Loads ``data/dataset/conspiracy_graph/manifest_prescreened_top.jsonl`` —
133 LLM-validated plausible attack chains — and exposes per-(objective,
role) score tables so :mod:`red_team.agent.tools.list_skills` can rank
skills by their evidence of being valid attack hosts instead of
returning a uniformly-random sample.

Each chain in the manifest carries:

* ``lineage.skill_chain``   — ordered skill ids forming the attack
* ``primary_attack_type``   — dataset-side label (e.g. ``credential_persistence``)
* ``rerank_score``          — 3-15 composite reranking score
* ``llm_score``             — 1-10 LLM plausibility score
* ``llm_verdict``           — ``plausible`` / ``implausible``
* ``nodes[].role``          — ``source`` / ``bridge`` / ``sink`` per skill

Score formula per skill in a matching chain:

    contribution = rerank_score
                 + 2 * llm_score (only if llm_verdict == "plausible")

A skill that appears in N matching chains accumulates N contributions.
The final ``score(skill | objective, role)`` is the sum.

The dataset's ``primary_attack_type`` does not 1:1 match the seed-side
``AttackObjective`` enum (paper §4 has 7 objectives; the dataset has
14 attack-type labels).  :data:`_ATTACK_TYPE_TO_OBJECTIVE` maps each
dataset label to its closest paper objective so callers using
``objective="privilege_escalation"`` see chains tagged
``command_injection`` too.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

__all__ = [
    "load_chain_priors",
    "rank_skills_for_objective",
    "ChainPriors",
]


# ---------------------------------------------------------------------------
# attack_type → AttackObjective mapping
# ---------------------------------------------------------------------------
# The conspiracy_graph dataset uses fine-grained "attack_type" labels
# (14 of them).  We fold them into the 7 paper objectives.  Where a
# dataset label spans multiple paper objectives, pick the dominant
# semantic.  Edge cases noted inline.

_ATTACK_TYPE_TO_OBJECTIVE: dict[str, str] = {
    # -- direct matches --------------------------------------------------
    "privilege_escalation":   "privilege_escalation",
    "data_exfiltration":      "data_exfiltration",

    # -- credential-related → data exfil (leaking creds == exfil) -------
    "credential_abuse":       "data_exfiltration",
    "repo_exfiltration":      "data_exfiltration",

    # -- persistence variants -------------------------------------------
    "credential_persistence": "persistence",
    "persistence_write":      "persistence",

    # -- destructive/poisoning → cleanup --------------------------------
    "data_poisoning":         "cleanup",

    # -- remote / supply-chain ------------------------------------------
    "remote_command_injection": "supply_chain_abuse",
    "remote_code_injection":    "supply_chain_abuse",
    "remote_code_execution":    "supply_chain_abuse",
    "remote_file_write":        "supply_chain_abuse",

    # -- local injection vectors → privesc (closest semantic) -----------
    "command_injection":      "privilege_escalation",
    "sql_injection":          "data_exfiltration",
    "remote_sql_injection":   "data_exfiltration",
}


# Walk up from src/red_team/agent/tools/_chain_priors.py to the repo root
# (4 parents: tools → agent → red_team → src → repo).
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MANIFEST = (
    _PROJECT_ROOT / "data" / "dataset"
    / "conspiracy_graph" / "manifest_prescreened_top.jsonl"
)


# ---------------------------------------------------------------------------
# Index structure
# ---------------------------------------------------------------------------


class ChainPriors:
    """Indexed scores from the prescreened-top chain manifest.

    Two indices: ``per_role[(objective, role)]`` and the role-agnostic
    ``per_any[objective]``.  Each maps to ``[(skill_id, score), ...]``
    sorted by score descending.

    Plus ``per_skill_breakdown[skill_id] = {objective: score, ...}`` so
    callers can show cross-objective scores for any skill.
    """

    __slots__ = (
        "per_role",
        "per_any",
        "per_skill_breakdown",
        "n_chains",
        "all_objectives",
    )

    def __init__(
        self,
        per_role: dict[tuple[str, str], list[tuple[str, float]]],
        per_any: dict[str, list[tuple[str, float]]],
        per_skill_breakdown: dict[str, dict[str, float]],
        n_chains: int,
    ) -> None:
        self.per_role = per_role
        self.per_any = per_any
        self.per_skill_breakdown = per_skill_breakdown
        self.n_chains = n_chains
        self.all_objectives = sorted(per_any.keys())

    def ranked_skills(
        self,
        objective: str,
        role: str = "any",
    ) -> list[tuple[str, float]]:
        """Return ``[(skill_id, score), ...]`` for the (objective, role) pair.

        ``role`` may be ``any`` / ``source`` / ``bridge`` / ``sink``.
        Empty list if no matching chains.
        """
        if role == "any":
            return list(self.per_any.get(objective, ()))
        return list(self.per_role.get((objective, role), ()))

    def ranked_skills_union(
        self,
        objectives: list[str],
        role: str = "any",
    ) -> list[tuple[str, float]]:
        """Union ranking across multiple objectives.

        For each skill, sum its scores from each requested objective and
        sort descending.  Used when the mutator wants to see "two-tier"
        skills that work across attack families (good for surface
        expansion or chain reuse).
        """
        if not objectives:
            return []
        merged: dict[str, float] = {}
        for obj in objectives:
            ranked = self.per_role.get((obj, role)) if role != "any" else self.per_any.get(obj)
            if not ranked:
                continue
            for skill_id, score in ranked:
                merged[skill_id] = merged.get(skill_id, 0.0) + score
        return sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))

    def cross_objective_count(self, skill_id: str) -> int:
        """How many distinct paper objectives this skill appears in."""
        return len(self.per_skill_breakdown.get(skill_id, {}))


def _score_for(chain: dict) -> float:
    """Per-chain contribution to each member skill's score."""
    rerank = float(chain.get("rerank_score", 0))
    llm_score = float(chain.get("llm_score", 0))
    bonus = 2.0 * llm_score if chain.get("llm_verdict") == "plausible" else 0.0
    return rerank + bonus


def _index_chains(manifest_lines: Iterable[str]) -> ChainPriors:
    per_role_acc: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    per_any_acc: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    n_chains = 0

    for line in manifest_lines:
        line = line.strip()
        if not line:
            continue
        try:
            chain = json.loads(line)
        except json.JSONDecodeError:
            continue
        attack_type = chain.get("primary_attack_type") or ""
        objective = _ATTACK_TYPE_TO_OBJECTIVE.get(attack_type)
        if objective is None:
            continue

        score = _score_for(chain)
        nodes = chain.get("nodes") or []
        skill_chain = (chain.get("lineage") or {}).get("skill_chain") or []

        # Build skill_id → role from the nodes block (node_id is
        # ``node:<skill_id>`` so we strip the prefix).
        skill_role: dict[str, str] = {}
        for node in nodes:
            nid = str(node.get("node_id") or "")
            role_str = str(node.get("role") or "any")
            if nid.startswith("node:"):
                skill_role[nid[len("node:"):]] = role_str

        for skill_id in skill_chain:
            if not skill_id:
                continue
            per_any_acc[objective][skill_id] += score
            role = skill_role.get(skill_id)
            if role:
                per_role_acc[(objective, role)][skill_id] += score

        n_chains += 1

    def _sorted(d: dict[str, float]) -> list[tuple[str, float]]:
        return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))

    per_role: dict[tuple[str, str], list[tuple[str, float]]] = {
        k: _sorted(v) for k, v in per_role_acc.items()
    }
    per_any: dict[str, list[tuple[str, float]]] = {
        k: _sorted(v) for k, v in per_any_acc.items()
    }
    # Per-skill breakdown: for each skill, which objectives it appears
    # in and with what score.  Used to surface cross-objective signal.
    per_skill_breakdown: dict[str, dict[str, float]] = {}
    for objective, skills in per_any_acc.items():
        for skill_id, score in skills.items():
            per_skill_breakdown.setdefault(skill_id, {})[objective] = score
    return ChainPriors(
        per_role=per_role,
        per_any=per_any,
        per_skill_breakdown=per_skill_breakdown,
        n_chains=n_chains,
    )


@lru_cache(maxsize=4)
def load_chain_priors(manifest_path: str | None = None) -> ChainPriors:
    """Load and index the prescreened-top chain manifest.

    Cached per-process — manifest is treated as static input; restart
    to pick up changes.  Empty index returned when manifest missing
    (so callers can degrade gracefully to the legacy random sample).
    """
    path = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST
    if not path.is_file():
        return ChainPriors({}, {}, {}, 0)
    with path.open("r", encoding="utf-8") as fh:
        return _index_chains(fh)


def rank_skills_for_objective(
    objective: str,
    *,
    role: str = "any",
    manifest_path: str | None = None,
) -> list[tuple[str, float]]:
    """Convenience wrapper: load priors and return the ranked list."""
    return load_chain_priors(manifest_path).ranked_skills(objective, role)
