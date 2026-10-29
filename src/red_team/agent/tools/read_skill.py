"""``read_skill`` tool — fetch a skill's SKILL.md + first entry script.

Benign split reads ``data/skills/clawhub/skills/<skill_id>/``; malicious
split scans ``data/skills/poisoned/*/<skill_id>/`` so the caller need
not know the ``chain_hash``.  Missing skill → structured error (not an
exception) so the LLM can self-correct on the next turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from red_team.agent.tools._data_root import (
    _data_root,
    _first_entry_script,
    _read_text_safely,
)
from red_team.agent.tools.base import _VALID_SKILL_SPLITS


_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": (
            "Read a skill's SKILL.md body plus the first 4KB of its "
            "entry script. Use this to imitate benign writing style or "
            "to inspect a previously-poisoned version of the current "
            "seed. No web fetch is performed — results come from the "
            "local marketplace mirror only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": (
                        "ClawHub skill directory name, e.g. 'tavily-web'."
                    ),
                },
                "split": {
                    "type": "string",
                    "enum": ["benign", "malicious"],
                    "default": "benign",
                    "description": (
                        "'benign' reads data/skills/clawhub/skills/; "
                        "'malicious' reads the previous round's poisoned "
                        "variant under data/skills/poisoned/<chain_hash>/."
                    ),
                },
            },
            "required": ["skill_id"],
        },
    },
}


@dataclass(frozen=True)
class ReadSkillTool:
    """Plugin implementation of ``read_skill``."""

    name: str = "read_skill"
    terminal: bool = False
    schema: dict = field(default_factory=lambda: _SCHEMA)

    def dispatch(self, args: dict, context: Any = None) -> dict[str, Any]:
        skill_id = args.get("skill_id")
        if not isinstance(skill_id, str) or not skill_id:
            return {"error": "skill_id is required (string)"}
        split = args.get("split", "benign")
        if split not in _VALID_SKILL_SPLITS:
            return {
                "error": (
                    f"invalid split {split!r}; expected one of "
                    f"{sorted(_VALID_SKILL_SPLITS)}"
                )
            }

        root = _data_root()

        if split == "benign":
            skill_dir = root / "skills" / "clawhub" / "skills" / skill_id
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.is_file():
                return {
                    "error": f"skill {skill_id!r} not found in benign split"
                }
            entry = _first_entry_script(skill_dir)
            payload: dict[str, Any] = {
                "skill_id": skill_id,
                "split": "benign",
                "skill_md": _read_text_safely(skill_md),
            }
            if entry is not None:
                # Paper spec: first 4KB of entry script.  We cap in bytes
                # then decode so "4KB" is an honest storage bound.
                try:
                    blob = entry.read_bytes()[:4096]
                    payload["entry_script_name"] = entry.name
                    payload["entry_script"] = blob.decode(
                        "utf-8", errors="replace"
                    )
                except OSError:
                    pass
            return payload

        # split == 'malicious'
        poisoned_root = root / "skills" / "poisoned"
        if poisoned_root.is_dir():
            # Sort so the first hit is deterministic across platforms.
            for chain_dir in sorted(poisoned_root.iterdir()):
                cand = chain_dir / skill_id / "SKILL.md"
                if cand.is_file():
                    return {
                        "skill_id": skill_id,
                        "split": "malicious",
                        "chain_hash": chain_dir.name,
                        "skill_md": _read_text_safely(cand),
                    }
        return {"error": "no malicious version"}
