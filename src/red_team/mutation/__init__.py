"""Red-team mutation module — L2 interface for "produce one variant" strategies.

This package owns the :class:`MutationStrategy` Protocol plus the two DTOs
(``MutationContext``, ``MutationResult``) shared across all strategies.
Evolution stages (L3) depend on this module; the reverse direction is
forbidden. Wave 3 只交付接口层骨架；后续 wave 会把 ``agent/runtime.py``
的 tool-use agent 和 ``baselines/`` 下的 random / zeroshot / blackbox
包成 :class:`MutationStrategy` 实现，搬进 ``mutation/strategies/``。
"""
from __future__ import annotations

from red_team.mutation.base import (
    MutationContext,
    MutationResult,
    MutationStrategy,
)
from red_team.mutation.search_space import (
    InvariantViolation,
    SearchSpace,
    shared_motif,
)

__all__ = [
    "MutationContext",
    "MutationResult",
    "MutationStrategy",
    "InvariantViolation",
    "SearchSpace",
    "shared_motif",
]
