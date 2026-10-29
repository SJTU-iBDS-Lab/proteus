"""Filesystem helpers shared by the per-tool plugins.

Everything here is deliberately stateless and side-effect free so the
tool plugins stay easy to unit-test.  The data-root is resolved
**per call** (not at import time) because test fixtures set the
``CLAW_DATA_ROOT`` env var inside pytest fixture bodies that run after
import.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from red_team.agent.tools.base import _PREVIEW_LIMIT, _ENTRY_SCRIPT_SUFFIXES


def _truncate(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    """Truncate *text* to *limit* chars, appending a single-char ellipsis.

    We use ``'…'`` (U+2026) rather than ``'...'`` because real SKILL.md
    bodies frequently end with trailing periods — the single-char marker
    is unambiguously the truncation signal.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _data_root() -> Path:
    """Return the data root, resolved per call.

    Precedence:
      1. ``CLAW_DATA_ROOT`` env var — used by tests to point at fixtures.
      2. ``<repo>/data`` relative to this file.  Walking four parents
         up lands at the repo root because this file is at
         ``src/red_team/agent/tools/_data_root.py``.
    """
    env = os.environ.get("CLAW_DATA_ROOT")
    if env:
        return Path(env).resolve()
    # _data_root.py -> tools -> agent -> red_team -> src -> <repo>
    return (Path(__file__).resolve().parents[4] / "data").resolve()


def _read_text_safely(path: Path, limit: int = _PREVIEW_LIMIT) -> str:
    """Read *path* as UTF-8 with a character cap, tolerating decode errors.

    Missing / unreadable files return an empty string — callers decide
    whether the absence is an error.  Decoding uses ``errors='replace'``
    so binary files don't crash the dispatcher.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover — decode=replace never raises
        return ""
    return _truncate(text, limit)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file; skip blank/comment lines.  Missing file → [].

    Non-dict rows and malformed JSON lines are silently dropped — the
    tools that consume these rows treat "unknown id" identically whether
    it is missing or malformed, so defensive tolerance here keeps the
    tool surface simple.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _first_entry_script(skill_dir: Path) -> Path | None:
    """Return the first entry script in *skill_dir* by preferred suffix.

    Preference order is :data:`_ENTRY_SCRIPT_SUFFIXES`; within a suffix
    we pick the lexicographically-first file so the "entry script" is
    stable across runs even when a skill ships multiple files.  Also
    probes a conventional ``scripts/`` sub-directory.
    """
    if not skill_dir.is_dir():
        return None
    for suffix in _ENTRY_SCRIPT_SUFFIXES:
        hits = sorted(skill_dir.glob(f"*{suffix}"))
        hits += sorted((skill_dir / "scripts").glob(f"*{suffix}"))
        if hits:
            return hits[0]
    return None
