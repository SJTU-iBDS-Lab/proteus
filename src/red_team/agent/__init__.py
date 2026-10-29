"""Red-team agent public API.

:data:`TOOLS`, :data:`TOOL_NAMES`, and :func:`_dispatch_tool` come from the
tools package; the prompt surface is in :mod:`red_team.agent.prompts`. Keep
the re-export list flat and explicit so downstream callers don't pull private
helpers by accident.
"""
from __future__ import annotations

from .prompts import (
    MUTATION_OPERATORS,
    ROUND_USER_TEMPLATE,
    SYSTEM_RED_TEAM,
    c_summary_from_variant,
    render_findings_block,
    render_round_user,
    search_space_rules_text,
    truncate_d,
)
from .tools import TOOL_NAMES, TOOLS, _dispatch_tool

__all__ = [
    "MUTATION_OPERATORS",
    "ROUND_USER_TEMPLATE",
    "SYSTEM_RED_TEAM",
    "TOOLS",
    "TOOL_NAMES",
    "_dispatch_tool",
    "c_summary_from_variant",
    "render_findings_block",
    "render_round_user",
    "search_space_rules_text",
    "truncate_d",
]
