"""A6 — ``search_space='evasion'``.

Maps directly to run_config.search_space.
"""
from __future__ import annotations

from red_team.campaign.run_config import RunConfig


def resolve_search_space(run_config: RunConfig) -> str:
    """Return the search space from the run config."""
    return run_config.search_space
