"""Tests for the ``red_team.agent.tools`` plugin base contract.

These tests pin the protocol each per-tool plugin must honour.  They run
independently of the legacy ``test_tools.py`` suite so a regression in
the split package shows up without being masked by the dispatcher-level
integration checks.
"""
from __future__ import annotations

import pytest

from red_team.agent.tools import base as tools_base


def test_protocol_surface_is_exposed() -> None:
    """``AgentTool`` must be importable from :mod:`red_team.agent.tools.base`
    and from the package root (re-export) so plugin authors have one
    canonical symbol, not two."""
    from red_team.agent.tools import AgentTool as pkg_level

    assert pkg_level is tools_base.AgentTool


def test_shared_constants_live_on_base() -> None:
    """Constants that used to sit in the monolithic ``tools.py`` now live
    on ``base`` so every plugin imports from one place.  The values are
    the frozen IMPL-01 contract — drift breaks prompt-token accounting."""
    assert tools_base._PREVIEW_LIMIT == 8000
    assert tools_base._BENIGN_SAMPLE_SIZE == 20
    assert tools_base._VALID_REPO_SECTIONS == frozenset(
        {"readme", "tree", "theme"}
    )
    assert tools_base._VALID_SKILL_SPLITS == frozenset(
        {"benign", "malicious"}
    )
    assert tools_base._ENTRY_SCRIPT_SUFFIXES == (".py", ".sh", ".js", ".ts")


def test_shared_constants_are_reexported_from_package_root() -> None:
    """Backward-compat: the old monolith module exposed these at the top
    level of ``red_team.agent.tools``; the split must preserve that so
    any external caller doing ``from red_team.agent.tools import
    _PREVIEW_LIMIT`` still works."""
    from red_team.agent import tools as pkg

    assert pkg._PREVIEW_LIMIT == tools_base._PREVIEW_LIMIT
    assert pkg._BENIGN_SAMPLE_SIZE == tools_base._BENIGN_SAMPLE_SIZE
    assert pkg._VALID_REPO_SECTIONS == tools_base._VALID_REPO_SECTIONS
    assert pkg._VALID_SKILL_SPLITS == tools_base._VALID_SKILL_SPLITS
    assert pkg._ENTRY_SCRIPT_SUFFIXES == tools_base._ENTRY_SCRIPT_SUFFIXES


def test_agent_tool_protocol_requires_schema_name_terminal() -> None:
    """A plugin that is missing any of the three required attrs must NOT
    pass ``isinstance(..., AgentTool)``.  We use a throwaway class so the
    failure mode is explicit rather than discovered in production."""

    class Incomplete:
        name = "x"
        terminal = False

        def dispatch(self, args, context=None):  # pragma: no cover
            return {}

    assert not isinstance(Incomplete(), tools_base.AgentTool)


def test_agent_tool_protocol_accepts_full_implementations() -> None:
    """A plugin that defines ``schema`` / ``name`` / ``terminal`` /
    ``dispatch`` must pass ``isinstance`` — otherwise the registry
    builder can't runtime-check new plugins."""

    class Ok:
        schema = {"type": "function", "function": {"name": "x"}}
        name = "x"
        terminal = False

        def dispatch(self, args, context=None):
            return {"ok": True}

    assert isinstance(Ok(), tools_base.AgentTool)
