"""A9 — ``feedback='scalar'``.

Maps to run_config.feedback_mode = "scalar".
"""
from __future__ import annotations

from typing import Any

from red_team.campaign.run_config import RunConfig


def filter_findings(findings: list[Any], run_config: RunConfig) -> list[Any]:
    """Return ``[]`` under scalar feedback, else the list unchanged."""
    if run_config.feedback_mode == "scalar":
        return []
    return list(findings)
