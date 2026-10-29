"""Programmatic baseline dispatch (extends A4 ``tool_use_off``).

Resolves the four ``agent_strategy`` values that bypass the
self-evolving agent loop and instead call a programmatic
:mod:`red_team.baselines` class:

    - "random_baseline"   → :class:`RandomMutation`
    - "llm_zeroshot"      → :class:`LLMZeroShot`
    - "blackbox_rewrite"  → :class:`BlackboxRewrite`

The fourth value, ``"llm"``, falls through to the agent loop and
returns ``None`` (caller dispatches to ``step_s1``).
"""
from __future__ import annotations

from red_team.campaign.run_config import RunConfig


# strategy → backend_tag (also used as the column label downstream)
_STRATEGY_TAGS: dict[str, str] = {
    "random_baseline": "random_baseline",
    "llm_zeroshot": "llm_zeroshot",
    "blackbox_rewrite": "blackbox_rewrite",
    "pyrit_standalone": "pyrit_standalone",
    "skill_inject_bo5": "skill_inject_bo5",
    "skillject_trace": "skillject_trace",
}


def try_short_circuit_agent(run_config: RunConfig) -> tuple[bool, str] | None:
    """If the strategy is a programmatic baseline, return
    ``(True, backend_tag)``; else return ``None``.

    The caller interprets the tuple as ``(short_circuit, backend_tag)``
    — when short-circuited it instantiates the baseline class
    corresponding to ``backend_tag`` and marks every resulting row with
    that tag.
    """
    tag = _STRATEGY_TAGS.get(run_config.agent_strategy)
    if tag is None:
        return None
    return True, tag
