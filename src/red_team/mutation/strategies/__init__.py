"""MutationStrategy adapters wrapping existing red-team engines.

Each adapter is a thin, frozen dataclass that maps
:class:`MutationContext` → the underlying engine's call signature and
repackages the ``AgentRunResult`` as a :class:`MutationResult`.
Evolution stages and campaign runners only ever depend on this registry;
they never import the concrete classes directly so new strategies
(tree-search, rule-based, …) can be added without touching call sites.
"""
from __future__ import annotations

from red_team.mutation.base import MutationStrategy
from red_team.mutation.strategies.blackbox import BlackboxStrategy
from red_team.mutation.strategies.random import RandomStrategy
from red_team.mutation.strategies.tool_use_agent import ToolUseAgentStrategy
from red_team.mutation.strategies.zeroshot import ZeroshotStrategy

REGISTRY: dict[str, MutationStrategy] = {  # type: ignore[dict-item]
    "tool_use_agent": ToolUseAgentStrategy(),
    "random": RandomStrategy(),
    "zeroshot": ZeroshotStrategy(),
    "blackbox": BlackboxStrategy(),
}


def get(name: str) -> MutationStrategy:
    """Return the registered strategy for ``name`` (KeyError if unknown)."""
    return REGISTRY[name]


__all__ = [
    "BlackboxStrategy",
    "RandomStrategy",
    "REGISTRY",
    "ToolUseAgentStrategy",
    "ZeroshotStrategy",
    "get",
]
