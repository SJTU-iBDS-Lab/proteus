"""Tests for the strategies REGISTRY — central lookup used by evolution stages."""
from __future__ import annotations

import pytest

from red_team.mutation.base import MutationStrategy


def test_registry_contains_four_keys() -> None:
    from red_team.mutation.strategies import REGISTRY

    assert set(REGISTRY.keys()) == {
        "tool_use_agent",
        "random",
        "zeroshot",
        "blackbox",
    }


def test_registry_values_satisfy_mutation_strategy_protocol() -> None:
    from red_team.mutation.strategies import REGISTRY

    for key, strat in REGISTRY.items():
        assert isinstance(strat, MutationStrategy), (
            f"registry entry {key!r} does not satisfy MutationStrategy"
        )


def test_registry_keys_match_strategy_names() -> None:
    """The key a caller uses to look up a strategy must equal its backend label."""
    from red_team.mutation.strategies import REGISTRY

    for key, strat in REGISTRY.items():
        assert strat.name == key, (
            f"registry key {key!r} != strategy.name {strat.name!r}"
        )


def test_get_returns_tool_use_agent_instance() -> None:
    from red_team.mutation.strategies import get
    from red_team.mutation.strategies.tool_use_agent import ToolUseAgentStrategy

    strat = get("tool_use_agent")
    assert isinstance(strat, ToolUseAgentStrategy)


def test_get_returns_random_instance() -> None:
    from red_team.mutation.strategies import get
    from red_team.mutation.strategies.random import RandomStrategy

    assert isinstance(get("random"), RandomStrategy)


def test_get_returns_zeroshot_instance() -> None:
    from red_team.mutation.strategies import get
    from red_team.mutation.strategies.zeroshot import ZeroshotStrategy

    assert isinstance(get("zeroshot"), ZeroshotStrategy)


def test_get_returns_blackbox_instance() -> None:
    from red_team.mutation.strategies import get
    from red_team.mutation.strategies.blackbox import BlackboxStrategy

    assert isinstance(get("blackbox"), BlackboxStrategy)


def test_get_unknown_key_raises_key_error() -> None:
    from red_team.mutation.strategies import get

    with pytest.raises(KeyError):
        get("tree_search")
