"""A5 — ``disable_expansion``.

Maps to run_config.expansion_enabled = False.
"""
from __future__ import annotations

from red_team.campaign.run_config import RunConfig


def is_expansion_disabled(run_config: RunConfig) -> bool:
    """True iff the run config disables horizontal expansion."""
    return not run_config.expansion_enabled
