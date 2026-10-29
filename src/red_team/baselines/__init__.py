"""Red-team baseline methods (paper §5.1).

This package hosts the Random / LLM zero-shot / Blackbox rewrite baselines
used by Table 3/4. Each baseline conforms to the :class:`RedTeamBaseline`
Protocol so the Part 3 driver can dispatch uniformly via ``--red-team``.
"""
from .base import (
    COST_PER_1K,
    RedTeamBaseline,
    estimate_cost,
    make_variant_from_new_d,
    sample_payload_fragment,
)

__all__ = [
    "COST_PER_1K",
    "RedTeamBaseline",
    "estimate_cost",
    "make_variant_from_new_d",
    "sample_payload_fragment",
]
