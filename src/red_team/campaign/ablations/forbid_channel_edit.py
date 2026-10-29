"""A8 — ``forbid_channel_edit``.

Maps to run_config.channel_edit_allowed = False.
"""
from __future__ import annotations

from typing import Any

from core.schemas import SkillVariant
from red_team.campaign.run_config import RunConfig
from red_team.evolution.search_space import InvariantViolation


def check_channel_invariant(
    parent: SkillVariant,
    child: SkillVariant,
    run_config: RunConfig,
) -> None:
    """Raise :class:`InvariantViolation` when kappa changed under the flag."""
    if not run_config.channel_edit_allowed:
        if set(parent.kappa) != set(child.kappa):
            raise InvariantViolation(
                "forbid_channel_edit violated — "
                f"parent.kappa={[k.value for k in parent.kappa]!r} "
                f"child.kappa={[k.value for k in child.kappa]!r}"
            )
