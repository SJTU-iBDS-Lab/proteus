"""``read_repo_context`` tool — return a section of a target repo card.

Loads ``data/targets/target_repos.jsonl`` (one JSON object per line).
Sections map to: ``readme``→``readme_md``, ``tree``→``file_tree``,
``theme``→``theme_summary``.  Missing / invalid inputs return a
structured error so the LLM can correct itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from red_team.agent.tools._data_root import (
    _data_root,
    _load_jsonl,
    _truncate,
)
from red_team.agent.tools.base import _VALID_REPO_SECTIONS


_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_repo_context",
        "description": (
            "Read one section of a target repository's context card "
            "(README, file tree, or theme summary) so mutations can "
            "sound natural in the repo's domain. Loads from "
            "data/targets/target_repos.jsonl."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "repo_id": {
                    "type": "string",
                    "description": (
                        "Target repo id, e.g. 'slopbuster'."
                    ),
                },
                "section": {
                    "type": "string",
                    "enum": ["readme", "tree", "theme"],
                    "description": (
                        "Which section of the repo card to return."
                    ),
                },
            },
            "required": ["repo_id", "section"],
        },
    },
}


@dataclass(frozen=True)
class ReadRepoContextTool:
    """Plugin implementation of ``read_repo_context``."""

    name: str = "read_repo_context"
    terminal: bool = False
    schema: dict = field(default_factory=lambda: _SCHEMA)

    def dispatch(self, args: dict, context: Any = None) -> dict[str, Any]:
        repo_id = args.get("repo_id")
        section = args.get("section")
        if not isinstance(repo_id, str) or not repo_id:
            return {"error": "repo_id is required (string)"}
        if section not in _VALID_REPO_SECTIONS:
            return {
                "error": (
                    f"invalid section {section!r}; expected one of "
                    f"{sorted(_VALID_REPO_SECTIONS)}"
                )
            }

        root = _data_root()
        rows = _load_jsonl(root / "targets" / "target_repos.jsonl")
        match = next(
            (r for r in rows if r.get("repo_id") == repo_id), None
        )
        if match is None:
            skill_dir = root / "skills" / "clawhub" / "skills" / repo_id
            skill_md = skill_dir / "SKILL.md"
            if not skill_dir.is_dir() or not skill_md.is_file():
                return {
                    "error": (
                        f"repo {repo_id!r} not found in target_repos.jsonl "
                        "or the downloaded ClawHub skills"
                    )
                }
            readme = skill_md.read_text(encoding="utf-8", errors="replace")
            inventory = sorted(
                str(path.relative_to(skill_dir))
                for path in skill_dir.rglob("*")
                if path.is_file() and ".git" not in path.parts
            )
            match = {
                "repo_id": repo_id,
                "readme_md": readme,
                "file_tree": inventory,
                # The frontmatter and opening prose are the closest analogue
                # to the short theme card used by the legacy ten repos.
                "theme_summary": readme[:2000],
            }

        field_map = {
            "readme": ("readme_md", str),
            "tree": ("file_tree", list),
            "theme": ("theme_summary", str),
        }
        key, expected_type = field_map[section]
        value = match.get(key, "" if expected_type is str else [])

        # Guard against giant READMEs / file trees.  Strings get the
        # preview cap; lists cap at 500 entries because some target repos
        # ship tens of thousands of paths.
        if isinstance(value, str):
            value = _truncate(value)
        elif isinstance(value, list):
            value = value[:500]

        return {"repo_id": repo_id, "section": section, "content": value}
