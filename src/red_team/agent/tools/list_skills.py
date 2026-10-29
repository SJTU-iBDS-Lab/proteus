"""``list_skills`` tool — sample marketplace skills + malicious seeds.

Two modes, switched by whether the caller supplies any of
``objective`` / ``objectives``:

* **Random sample** (legacy, no objective set) — returns a
  deterministic 20-skill sample from ``data/skills/clawhub/skills/``.
  Kept for backward-compat with prompts/tests that don't yet know
  about the prior-ranking mode.

* **Prior-ranked** — ranks ClawHub skills by evidence in
  ``data/dataset/conspiracy_graph/manifest_prescreened_top.jsonl``
  (133 LLM-validated plausible attack chains).  Each chain contributes
  ``rerank_score + 2*llm_score`` to every member skill matching the
  objective; results are returned sorted by score desc.  Three knobs:

  - ``objective``  — single paper objective (e.g.
    ``privilege_escalation``).
  - ``objectives`` — list of paper objectives, scores summed across
    them so cross-family-strong skills float to the top (used for
    surface expansion + chain reuse).  Pass ``"all"`` to union all
    seven paper objectives.
  - ``role``       — any | source | bridge | sink.

  ``page`` / ``limit`` page through the ranking.  When the page
  exceeds ``total_matched``, the response surfaces a ``guidance``
  block hinting at next-step queries (different role, different
  objective combo, or ``exclude`` list) — there is **no fallback to
  a lower-precision pool**; the LLM decides explicitly what to try
  next.

Each item exposes its ``score_breakdown`` (per-objective scores) so
the LLM sees a skill's full attack-family footprint, not just its
score in the current query.

``malicious`` always returns the curated seeds from
``data/seeds/malicious_seeds.jsonl`` (small, no ranking needed).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from red_team.agent.tools._chain_priors import load_chain_priors
from red_team.agent.tools._data_root import (
    _data_root,
    _load_jsonl,
    _truncate,
)
from red_team.agent.tools.base import _BENIGN_SAMPLE_SIZE


_VALID_ROLES = ("any", "source", "bridge", "sink")
_VALID_OBJECTIVES = (
    "data_exfiltration",
    "persistence",
    "privilege_escalation",
    "cleanup",
    "guidance_injection",
    "supply_chain_abuse",
    "cross_skill_chaining",
)


_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_skills",
        "description": (
            "List marketplace skills + malicious seeds.\n\n"
            "Without `objective`/`objectives`: returns a "
            "deterministic 20-skill sample (random across alphabet, "
            "NOT relevance-ranked — calling this twice gives the "
            "same result; do not waste rounds on duplicate calls).\n\n"
            "With `objective` (single) or `objectives` (list/all): "
            "ranks ClawHub skills by evidence in 133 LLM-validated "
            "attack chains.  Each chain contributes "
            "`rerank_score + 2*llm_score` (if plausible) to its "
            "member skills.  Items show `score_breakdown` per "
            "objective so you can spot cross-family-strong skills.\n\n"
            "When you know the seed's `o`, ALWAYS call with "
            "`objective=<o>` first.  Then explore neighbours via:\n"
            "  - `role='source'/'sink'/'bridge'` for chain-position-"
            "specific candidates;\n"
            "  - `objectives=['o1','o2',...]` to UNION rankings "
            "across families (good for surface_expansion or chain "
            "reuse);\n"
            "  - `objectives='all'` to see which skills are strong "
            "globally;\n"
            "  - `page=N+1` to walk past the top-20.\n\n"
            "When `total_matched < (page+1)*limit` the response "
            "surfaces a `guidance` block with concrete next queries; "
            "there is NO automatic fallback to a lower-quality pool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["benign", "malicious", "all"],
                    "default": "benign",
                    "description": "Which corpus to sample from.",
                },
                "objective": {
                    "type": "string",
                    "enum": list(_VALID_OBJECTIVES),
                    "description": (
                        "Single paper-aligned attack objective.  "
                        "Match this to the seed's `o` field for the "
                        "primary query.  Mutually exclusive with "
                        "`objectives`."
                    ),
                },
                "objectives": {
                    "oneOf": [
                        {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(_VALID_OBJECTIVES),
                            },
                            "minItems": 1,
                        },
                        {"type": "string", "enum": ["all"]},
                    ],
                    "description": (
                        "List of paper objectives whose rankings "
                        "should be UNIONed (sum of scores per "
                        "skill).  Pass 'all' to union all seven.  "
                        "Use this to find cross-family-strong "
                        "skills for surface_expansion / chain reuse.  "
                        "Mutually exclusive with `objective`."
                    ),
                },
                "role": {
                    "type": "string",
                    "enum": list(_VALID_ROLES),
                    "default": "any",
                    "description": (
                        "Filter by chain role: source = attack "
                        "entry-point, sink = final payload, bridge "
                        "= intermediate hop.  Only meaningful with "
                        "`objective`/`objectives` set."
                    ),
                },
                "page": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": (
                        "Zero-indexed page for pagination.  When "
                        "exhausted, response includes a `guidance` "
                        "block hinting at next-step queries."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                    "description": (
                        "Max items per page (1-50)."
                    ),
                },
                "exclude": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of skill_ids to filter out of the "
                        "ranking (e.g., ones already tried this "
                        "campaign).  Only the unfiltered remainder "
                        "is paged."
                    ),
                },
            },
            "required": ["category"],
        },
    },
}


def _extract_benign_one_liner(skill_md: Path) -> str:
    """Pull the ``description:`` line out of a SKILL.md front-matter.

    SKILL.md format is YAML front-matter followed by Markdown body.
    We do line-based string slicing (no YAML dependency) because the
    ClawHub mirror always keeps description on a single line, and
    pulling in PyYAML just for this one field would be disproportionate.
    """
    if not skill_md.is_file():
        return ""
    try:
        with skill_md.open("r", encoding="utf-8", errors="replace") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith("description:"):
                    desc = stripped[len("description:"):].strip()
                    if (
                        len(desc) >= 2
                        and desc[0] in {'"', "'"}
                        and desc[-1] == desc[0]
                    ):
                        desc = desc[1:-1]
                    return _truncate(desc, 300)
    except OSError:
        return ""
    return ""


def _build_guidance(
    priors: Any,
    objectives: list[str],
    role: str,
    total: int,
    page: int,
    limit: int,
) -> list[str]:
    """Concrete next-query hints when the current page is exhausted."""
    hints: list[str] = ["no more results in current query"]

    # Suggest other roles for the same objectives.
    if role != "any":
        for alt_role in ("source", "sink", "bridge", "any"):
            if alt_role == role:
                continue
            alt_total = sum(
                len(priors.per_role.get((o, alt_role), []))
                for o in objectives
                if alt_role != "any"
            )
            if alt_role == "any":
                alt_total = sum(len(priors.per_any.get(o, [])) for o in objectives)
            if alt_total > 0:
                hints.append(
                    f"try role='{alt_role}' (same objectives) — "
                    f"{alt_total} candidates"
                )
                break

    # Suggest unioning with a sibling objective.
    if len(objectives) == 1:
        # Find a sibling objective with the most non-overlapping skills.
        current = set(s for s, _ in (priors.per_any.get(objectives[0]) or []))
        best: tuple[str, int] | None = None
        for sib in priors.all_objectives:
            if sib == objectives[0]:
                continue
            sib_skills = set(s for s, _ in (priors.per_any.get(sib) or []))
            new_count = len(sib_skills - current)
            if best is None or new_count > best[1]:
                best = (sib, new_count)
        if best and best[1] > 0:
            hints.append(
                f"try objectives=['{objectives[0]}','{best[0]}'] — "
                f"adds {best[1]} new candidates"
            )

    hints.append(
        "or pass `exclude=[ids you've tried]` and refresh the same query"
    )
    return hints


def _ranked_benign_items(
    objectives: list[str],
    role: str,
    page: int,
    limit: int,
    exclude: set[str],
    clawhub_root: Path,
) -> dict[str, Any]:
    """Build the ``benign`` items list using chain-prior ranking.

    ``objectives`` is always a list (single-objective queries pass a
    1-element list).  Score is the union sum across all listed
    objectives; ``score_breakdown`` per item exposes the per-objective
    contributions so the LLM sees a skill's full footprint.
    """
    priors = load_chain_priors()
    if len(objectives) == 1:
        ranked = priors.ranked_skills(objectives[0], role)
    else:
        ranked = priors.ranked_skills_union(objectives, role)

    # Apply exclude filter before paging so the caller sees a stable
    # view (exclude=[X] always removes X regardless of page).
    if exclude:
        ranked = [(s, sc) for s, sc in ranked if s not in exclude]

    total = len(ranked)
    start = page * limit
    end = start + limit
    page_slice = ranked[start:end]

    items: list[dict[str, Any]] = []
    for skill_id, score in page_slice:
        skill_dir = clawhub_root / skill_id
        one_liner = _extract_benign_one_liner(skill_dir / "SKILL.md")
        breakdown_raw = priors.per_skill_breakdown.get(skill_id, {})
        breakdown = {
            obj: round(float(sc), 2) for obj, sc in breakdown_raw.items()
        }
        items.append({
            "id": skill_id,
            "split": "benign",
            "one_liner": one_liner,
            "score": round(float(score), 2),
            "score_breakdown": breakdown,
            "appears_in_objectives": len(breakdown),
        })

    ranking: dict[str, Any] = {
        "objectives": objectives,
        "role": role,
        "page": page,
        "limit": limit,
        "total_matched": total,
        "has_more": end < total,
        "exhausted": end >= total,
        "exclude_count": len(exclude),
        "source": "manifest_prescreened_top.jsonl (133 chains)",
    }
    if end >= total:
        ranking["guidance"] = _build_guidance(
            priors, objectives, role, total, page, limit
        )
    return {"items": items, "ranking": ranking}


def _random_benign_items(clawhub_root: Path) -> list[dict[str, Any]]:
    """Legacy deterministic 20-skill random sample."""
    if not clawhub_root.is_dir():
        return []
    all_dirs = sorted(p for p in clawhub_root.iterdir() if p.is_dir())
    rng = random.Random(f"list_skills::benign::{len(all_dirs)}")
    sample = (
        rng.sample(all_dirs, _BENIGN_SAMPLE_SIZE)
        if len(all_dirs) > _BENIGN_SAMPLE_SIZE
        else all_dirs
    )
    sample.sort(key=lambda p: p.name)
    items: list[dict[str, Any]] = []
    for p in sample:
        one_liner = _extract_benign_one_liner(p / "SKILL.md")
        items.append({
            "id": p.name,
            "split": "benign",
            "one_liner": one_liner,
        })
    return items


@dataclass(frozen=True)
class ListSkillsTool:
    """Plugin implementation of ``list_skills``."""

    name: str = "list_skills"
    terminal: bool = False
    schema: dict = field(default_factory=lambda: _SCHEMA)

    def dispatch(self, args: dict, context: Any = None) -> dict[str, Any]:
        category = args.get("category", "benign")
        if category not in {"benign", "malicious", "all"}:
            return {
                "error": (
                    f"invalid category {category!r}; expected "
                    f"benign|malicious|all"
                )
            }

        # Normalise objective / objectives → single ``objectives`` list.
        objective = args.get("objective")
        objectives_arg = args.get("objectives")
        if objective is not None and objectives_arg is not None:
            return {
                "error": (
                    "pass either `objective` (single) or `objectives` "
                    "(list/'all'), not both"
                )
            }
        objectives: list[str] = []
        if objective is not None:
            if objective not in _VALID_OBJECTIVES:
                return {
                    "error": (
                        f"invalid objective {objective!r}; expected "
                        f"{list(_VALID_OBJECTIVES)}"
                    )
                }
            objectives = [objective]
        elif objectives_arg is not None:
            if objectives_arg == "all":
                objectives = list(_VALID_OBJECTIVES)
            elif isinstance(objectives_arg, list):
                bad = [o for o in objectives_arg if o not in _VALID_OBJECTIVES]
                if bad:
                    return {
                        "error": (
                            f"invalid objectives {bad!r}; expected each "
                            f"of {list(_VALID_OBJECTIVES)} or string 'all'"
                        )
                    }
                objectives = list(objectives_arg)
            else:
                return {
                    "error": (
                        "`objectives` must be a list of objective strings "
                        "or the literal string 'all'"
                    )
                }

        role = args.get("role", "any")
        if role not in _VALID_ROLES:
            return {"error": f"invalid role {role!r}; expected {list(_VALID_ROLES)}"}

        page = max(0, int(args.get("page", 0) or 0))
        limit = max(1, min(50, int(args.get("limit", 20) or 20)))
        exclude_arg = args.get("exclude") or []
        if not isinstance(exclude_arg, list):
            return {"error": "`exclude` must be a list of skill_ids"}
        exclude_set = {str(x) for x in exclude_arg}

        root = _data_root()
        result: dict[str, Any] = {"category": category, "items": []}

        if category in {"benign", "all"}:
            clawhub = root / "skills" / "clawhub" / "skills"
            if objectives:
                ranked = _ranked_benign_items(
                    objectives, role, page, limit, exclude_set, clawhub
                )
                result["items"].extend(ranked["items"])
                result["ranking"] = ranked["ranking"]
            else:
                result["items"].extend(_random_benign_items(clawhub))

        if category in {"malicious", "all"}:
            seeds = _load_jsonl(root / "seeds" / "malicious_seeds.jsonl")
            for row in seeds:
                result["items"].append({
                    "id": row.get("seed_id", "<unknown>"),
                    "split": "malicious",
                    "objective": row.get("o", ""),
                    "chain": row.get("chain", []),
                })

        result["count"] = len(result["items"])
        return result
