"""Ablation hooks consumed by :mod:`red_team.campaign.runner`.

Each hook is a small pure function addressing exactly one knob from the
paper's ablation matrix (A1…A9).  The runner calls them at specific
points in the state machine; none of them own state and none of them
know about each other.

Registry shape
--------------

Hooks are grouped by *phase* so the runner does not carry a long
if/elif chain over the ablation dict:

* ``feedback_hooks``        — run on ``(findings, run_config) → findings``
  before ``agent_run`` (only A9 subscribes today).
* ``agent_dispatch_hooks``  — run on ``(run_config) → backend_tag | None``
  to decide whether to short-circuit the agent (only A4 subscribes).
* ``mutation_invariants``   — run on ``(parent, child, run_config) → None``
  and raise :class:`red_team.evolution.search_space.InvariantViolation`
  when a flag's post-mutation invariant is violated (A8 subscribes).
* ``sandbox_rewriters``     — run on ``(variant, run_config) → (variant',
  forced_flag, chain_original, chain_used)`` to rewrite the variant that
  is handed to the sandbox (A7 subscribes).
* ``expansion_gates``       — run on ``(run_config) → bool`` and return
  ``False`` to skip horizontal expansion (A5 subscribes).

Adding a new ablation
---------------------

1. Create ``ablations/<name>.py`` with the hook function(s).
2. Register it in the list below (or in a new phase list if needed).
3. Add a unit test in ``tests/unit/red_team/campaign/`` that exercises
   the hook in isolation; never add ``if ablation == "<name>"`` to the
   runner.

A1 / A2 / A3 / A6 are runtime-side (``disable_tool`` / ``search_space``)
and flow to ``agent_run`` verbatim via ``runner`` — no hook is needed
because the driver merely forwards them.
"""
from __future__ import annotations

from typing import Any, Callable

from core.schemas import SkillVariant

from red_team.campaign.run_config import RunConfig
from red_team.campaign.ablations.disable_expansion import (
    is_expansion_disabled as _is_expansion_disabled,
)
from red_team.campaign.ablations.forbid_channel_edit import (
    check_channel_invariant as _check_channel_invariant,
)
from red_team.campaign.ablations.force_topology_single import (
    maybe_force_single_topology as _maybe_force_single_topology,
)
from red_team.campaign.ablations.scalar_feedback import (
    filter_findings as _filter_findings,
)
from red_team.campaign.ablations.tool_use_off import (
    try_short_circuit_agent as _try_short_circuit_agent,
)

__all__ = [
    "apply_feedback_hooks",
    "select_agent_dispatch",
    "check_mutation_invariants",
    "apply_sandbox_rewriters",
    "is_expansion_enabled",
]


# Each phase is a list of callables so adding a new ablation = append
# a new entry here + drop a module in this package.  The runner never
# inspects ablation names directly.

_FeedbackHook = Callable[[list[Any], RunConfig], list[Any]]
_AgentDispatchHook = Callable[[RunConfig], tuple[bool, Any] | None]
_MutationInvariant = Callable[[SkillVariant, SkillVariant, RunConfig], None]
_SandboxRewriter = Callable[
    [SkillVariant, RunConfig],
    tuple[SkillVariant, bool, list[str] | None, list[str] | None],
]
_ExpansionGate = Callable[[RunConfig], bool]


_FEEDBACK_HOOKS: list[_FeedbackHook] = [_filter_findings]
_MUTATION_INVARIANTS: list[_MutationInvariant] = [_check_channel_invariant]
_SANDBOX_REWRITERS: list[_SandboxRewriter] = [_maybe_force_single_topology]


def apply_feedback_hooks(
    findings: list[Any], run_config: RunConfig
) -> list[Any]:
    """Run every feedback hook in order.  Each hook may return a new list."""
    out = list(findings)
    for hook in _FEEDBACK_HOOKS:
        out = hook(out, run_config)
    return out


def select_agent_dispatch(run_config: RunConfig) -> tuple[bool, Any] | None:
    """Return ``(short_circuit, payload)`` if an ablation swaps the agent.

    Returns ``None`` to mean "dispatch to the real agent loop".  Today
    only A4 (``tool_use_off``) short-circuits — extend the
    ``tool_use_off`` module if other flags ever need the same seam.
    """
    return _try_short_circuit_agent(run_config)


def check_mutation_invariants(
    parent: SkillVariant,
    child: SkillVariant,
    run_config: RunConfig,
) -> None:
    """Run every mutation invariant.  May raise :class:`InvariantViolation`."""
    for hook in _MUTATION_INVARIANTS:
        hook(parent, child, run_config)


def apply_sandbox_rewriters(
    variant: SkillVariant, run_config: RunConfig
) -> tuple[SkillVariant, bool, list[str] | None, list[str] | None]:
    """Run every sandbox rewriter.

    Today only A7 rewrites the variant, so the returned ``(forced,
    chain_original, chain_used)`` triple comes from the first rewriter
    that claims the variant (``forced == True``).  If none claim it,
    the original variant is returned untouched.
    """
    for hook in _SANDBOX_REWRITERS:
        out = hook(variant, run_config)
        if out[1]:  # forced_flag
            return out
    return variant, False, None, None


def is_expansion_enabled(run_config: RunConfig) -> bool:
    """Return ``False`` iff any ablation disables horizontal expansion."""
    return not _is_expansion_disabled(run_config)
