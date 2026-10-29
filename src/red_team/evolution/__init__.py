"""Red-team evolution driver (paper §4.3, Algorithm 1).

This package hosts the three-layer evolution loop:

* ``S1`` evasion  — only the documentation text ``d`` may change.
* ``S2`` path expansion — ``c``, ``κ``, ``d`` (and chain within τ) may change.
* ``S3`` surface expansion — any field may change, provided the child still
  shares a structural motif with the parent.

:class:`SearchSpace` encodes the mutation mask and invariants enforced at each
level; :class:`InvariantViolation` is raised when a child mutation escapes the
allowed space.  :func:`shared_motif` implements the S3 acceptance gate.
"""
from __future__ import annotations

from .search_space import InvariantViolation, SearchSpace, shared_motif

__all__ = [
    "InvariantViolation",
    "SearchSpace",
    "shared_motif",
]
