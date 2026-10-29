"""Tests for the per-tool plugin registry assembled by the package
``__init__.py``.

Scope of this suite:
  * Every registered tool instance honours :class:`AgentTool`.
  * Registry ordering matches the frozen ``TOOL_NAMES`` contract.
  * ``submit_mutation`` is the only terminal tool.
  * The forbidden ``web_search`` / ``browse`` tools never leak in.
  * ``_dispatch_tool`` is module-level on the package root (monkeypatch
    target the runtime and tests rely on).
"""
from __future__ import annotations

import pytest

from red_team.agent import tools as tools_pkg
from red_team.agent.tools import base as tools_base


def test_registry_contains_six_instances() -> None:
    """Same count as the frozen schema list; drift should fail loudly."""
    assert len(tools_pkg._TOOL_INSTANCES) == 6


def test_registry_order_matches_tool_names() -> None:
    """Prompt templates (IMPL-02) enumerate tools positionally, so order
    is part of the public contract.  The registry, ``TOOL_NAMES``, and
    ``TOOLS`` must all agree."""
    names_from_registry = tuple(t.name for t in tools_pkg._TOOL_INSTANCES)
    assert names_from_registry == tools_pkg.TOOL_NAMES
    schema_names = tuple(t["function"]["name"] for t in tools_pkg.TOOLS)
    assert schema_names == tools_pkg.TOOL_NAMES


def test_registry_entries_are_agent_tool_instances() -> None:
    for t in tools_pkg._TOOL_INSTANCES:
        assert isinstance(t, tools_base.AgentTool)


def test_only_submit_mutation_is_terminal() -> None:
    """Terminal sentinel is how the runtime knows to stop the loop.
    Adding another terminal tool silently would let the agent bypass
    validation by committing via the wrong tool."""
    terminals = [t.name for t in tools_pkg._TOOL_INSTANCES if t.terminal]
    assert terminals == ["submit_mutation"]


def test_no_web_search_in_registry() -> None:
    """Paper §4.2 forbids public-web evidence; enforcing it on the
    registry rather than the prompt keeps an adversarial runtime
    honest."""
    names = {t.name for t in tools_pkg._TOOL_INSTANCES}
    assert "web_search" not in names
    assert "browse" not in names
    assert "fetch_url" not in names


def test_dispatch_tool_is_module_level_on_package() -> None:
    """``runtime.py`` imports ``_dispatch_tool`` directly and tests rely
    on ``monkeypatch.setattr(tools, '_dispatch_tool', ...)``; exposing
    it anywhere else would silently break those callers."""
    assert hasattr(tools_pkg, "_dispatch_tool")
    assert callable(tools_pkg._dispatch_tool)


def test_dispatch_submit_mutation_returns_terminal_sentinel_via_package() -> None:
    """Smoke-check that the dispatcher survives the split: the sentinel
    contract ``{"_terminal": True, "args": ...}`` is what IMPL-05 keys on."""
    args = {"new_d": "x", "rationale": "y"}
    result = tools_pkg._dispatch_tool(
        "submit_mutation", args, None, None, None, None
    )
    assert result["_terminal"] is True
    assert result["args"] == args
    # Defensive copy: mutating the sentinel must not touch the caller's dict.
    result["args"]["new_d"] = "mutated"
    assert args["new_d"] == "x"


def test_dispatch_unknown_tool_raises_keyerror_via_package() -> None:
    with pytest.raises(KeyError):
        tools_pkg._dispatch_tool("no_such_tool", {}, None, None, None, None)


def test_tools_is_a_tuple_of_schemas() -> None:
    """Tuple (not list) prevents accidental mutation of the frozen
    contract by downstream code.  The legacy module exposed a list;
    we tighten to tuple because the schema object is effectively
    immutable public API."""
    assert isinstance(tools_pkg.TOOLS, tuple)
    assert isinstance(tools_pkg.TOOL_NAMES, tuple)


def test_package_all_reexports_public_surface() -> None:
    """The ``__all__`` list is what ``from red_team.agent.tools import *``
    resolves to; it must cover every symbol the legacy monolith exposed."""
    for name in (
        "TOOLS",
        "TOOL_NAMES",
        "_dispatch_tool",
        "AgentTool",
        "_PREVIEW_LIMIT",
        "_BENIGN_SAMPLE_SIZE",
        "_VALID_REPO_SECTIONS",
        "_VALID_SKILL_SPLITS",
        "_ENTRY_SCRIPT_SUFFIXES",
    ):
        assert name in tools_pkg.__all__, name
        assert hasattr(tools_pkg, name), name
