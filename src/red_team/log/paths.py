"""Single source of truth for ``outputs/evolution/<run_id>/...`` layout.

Every red-team log writer must resolve filesystem targets through the
helpers defined here. Hand-rolled ``os.path.join`` or f-string path
concatenation is forbidden because it silently drifts when the layout
changes.

Layout
------
::

    <root>/<run_id>/
        log.jsonl          # campaign rows
        attack_log.jsonl   # cross-round BM25 memory
        summary.json       # terminal summary
        mutations/         # per-round SkillVariant materializations
"""
from __future__ import annotations

from pathlib import Path

_DEFAULT_OUTPUT_ROOT = Path("outputs/evolution")


def run_root(run_id: str, root: Path | None = None) -> Path:
    """Return ``<root>/<run_id>/`` (default root = ``outputs/evolution``)."""
    return (root or _DEFAULT_OUTPUT_ROOT) / run_id


def campaign_log_path(run_id: str, root: Path | None = None) -> Path:
    """Return the absolute-ish path of ``log.jsonl`` for a run."""
    return run_root(run_id, root) / "log.jsonl"


def attack_log_path(run_id: str, root: Path | None = None) -> Path:
    """Return the path of ``attack_log.jsonl`` for a run."""
    return run_root(run_id, root) / "attack_log.jsonl"


def summary_path(run_id: str, root: Path | None = None) -> Path:
    """Return the path of ``summary.json`` for a run."""
    return run_root(run_id, root) / "summary.json"


def mutation_dir(run_id: str, root: Path | None = None) -> Path:
    """Return the directory holding per-round SkillVariant materializations."""
    return run_root(run_id, root) / "mutations"


__all__ = [
    "run_root",
    "campaign_log_path",
    "attack_log_path",
    "summary_path",
    "mutation_dir",
]
