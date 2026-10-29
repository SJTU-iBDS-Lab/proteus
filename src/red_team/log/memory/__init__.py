"""Run-scope memory subsystem (Phase 2, Medium scope).

Re-exports the single immutable DTO :class:`MemoryUnit` (no edges in
the Medium plan — evolution chains are tracked by the
``superseded_by`` field on the unit itself).

The HarvestRow extractor input lives one level up at
:mod:`red_team.log.harvest`.
"""
from __future__ import annotations

from red_team.log.memory.unit import MemoryUnit

__all__ = ["MemoryUnit"]
