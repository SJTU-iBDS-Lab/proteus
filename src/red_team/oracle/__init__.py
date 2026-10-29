"""Red-team L1 lethality oracle.

Exports:
  * :func:`lethal` — back-compat bool-returning predicate.
  * :class:`ObjectiveJudge` — per-objective protocol.
  * :data:`REGISTRY` — :class:`AttackObjective` → :class:`ObjectiveJudge`.
"""
from .base import ObjectiveJudge
from .lethality import lethal
from .objectives import REGISTRY

__all__ = ["ObjectiveJudge", "REGISTRY", "lethal"]
