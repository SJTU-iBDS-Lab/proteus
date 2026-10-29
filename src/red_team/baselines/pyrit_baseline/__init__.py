"""PyRIT baseline plugin (Microsoft Python Risk Identification Tool).

This package provides two integration modes that share a single adapter
layer.  Nothing in this module mutates the rest of the codebase at
import time; every entry point is opt-in:

  * ``standalone.PyRITStandalone``
        ``RedTeamBaseline`` Protocol implementation.  PyRIT generates a
        jailbreak prompt; we wrap it as a SkillVariant by reusing the
        parent's ``c``/``kappa``/``chain`` and only swapping ``d``.
        This is the "tool only, no Proteus loop" arm of the comparison.

  * ``agent_tool.maybe_install``
        Optional runtime-injection of a 7th agent tool ``pyrit_generate``
        gated by the ``PYRIT_TOOL=1`` environment variable.  When unset
        (the default), the canonical 6-tool contract is preserved bit-for-bit.
        This is the "Proteus + PyRIT tool" arm.

  * ``litellm_target.build_litellm_chat_target``
        Adapter that wires PyRIT's ``OpenAIChatTarget`` to the existing
        LiteLLM proxy at ``http://127.0.0.1:4000`` so PyRIT shares the
        same model endpoints as Proteus runs.

  * ``prompt_to_variant.wrap_prompt_as_variant``
        Maps a PyRIT-generated jailbreak prompt to a ``SkillVariant``
        suitable for the audit→sandbox→oracle pipeline.

Nothing here is imported by core Proteus paths; importing this module
has no side effect on existing tests or runs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from red_team.baselines.pyrit_baseline.litellm_target import (
        build_litellm_chat_target,
    )
    from red_team.baselines.pyrit_baseline.prompt_to_variant import (
        wrap_prompt_as_variant,
    )
    from red_team.baselines.pyrit_baseline.standalone import PyRITStandalone

__all__ = [
    "PyRITStandalone",
    "build_litellm_chat_target",
    "wrap_prompt_as_variant",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy import bookkeeping
    if name == "PyRITStandalone":
        from red_team.baselines.pyrit_baseline.standalone import PyRITStandalone
        return PyRITStandalone
    if name == "build_litellm_chat_target":
        from red_team.baselines.pyrit_baseline.litellm_target import build_litellm_chat_target
        return build_litellm_chat_target
    if name == "wrap_prompt_as_variant":
        from red_team.baselines.pyrit_baseline.prompt_to_variant import wrap_prompt_as_variant
        return wrap_prompt_as_variant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
