"""Backward-compatibility shim — search_space has moved to ``red_team.mutation``.

The authoritative module is now :mod:`red_team.mutation.search_space`.
This file re-exports the public surface so legacy imports continue to
work (``from red_team.evolution.search_space import SearchSpace, ...``)
without forcing every downstream caller to update in the same commit.

New code MUST import from :mod:`red_team.mutation.search_space` directly.
"""
from __future__ import annotations

from red_team.mutation.search_space import (
    InvariantViolation,
    SearchSpace,
    _FROZEN,
    _LEGACY_LEVELS,
    _MUTABLE,
    _VALID_LEVELS,
    _normalize_level,
    shared_motif,
)

__all__ = [
    "InvariantViolation",
    "SearchSpace",
    "shared_motif",
    "_LEGACY_LEVELS",
    "_VALID_LEVELS",
    "_FROZEN",
    "_MUTABLE",
    "_normalize_level",
]
