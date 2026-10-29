"""Evasion evolution step — thin forward to :func:`agent_run`.

Paper §4.3 / Algorithm 1 defines three evolution levels — in paper
notation ``S1`` / ``S2`` / ``S3``, mapped to the semantic names
``evasion`` / ``path_expansion`` / ``surface_expansion`` — that differ
only in which axes of the ``(o, τ, c, κ, d, chain)`` factorization may
be mutated.  For the ``evasion`` level, *only* the documentation body
``d`` is mutable — every other axis is frozen against the parent.

This module is intentionally minimal: the search-space invariants are
already policed by :class:`red_team.evolution.search_space.SearchSpace`
and the underlying LLM loop lives in
:func:`red_team.agent.runtime.agent_run`.  This module therefore exists
solely as a named entry point for the evolution driver.

The function signature takes ``agent_run`` as a keyword-only injected
callable: that keeps unit tests hermetic (they can pass a ``Mock``)
without reaching for monkey-patching, and it lets the driver
swap in a stubbed runner for dry-run / replay flows.

Contract (paper §4.3 → search_space.SearchSpace(level="evasion")):

* ``memory_enabled``     — pulled from ``run_config.memory_enabled``
* ``attack_log_path``    — forwarded verbatim; the caller decides whether
  to persist a per-round log.
* ``run_config`` + ``round_index`` — forwarded unchanged.
* Return value           — the :class:`AgentRunResult` produced by
  ``agent_run`` is returned verbatim, with no post-processing.  Any
  exception raised by ``agent_run`` propagates to the caller.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from core.models import RepositorySlice
from core.schemas import PaperFinding, SkillVariant
from red_team.baselines.base import AgentRunResult
from red_team.campaign.run_config import RunConfig


def step_s1(
    parent: SkillVariant,
    findings: list[PaperFinding],
    repo: RepositorySlice,
    *,
    agent_run: Callable[..., AgentRunResult],
    run_config: RunConfig,
    round_index: int,
    attack_log_path: Path | None = None,
    search_space: str = "evasion",
    last_round_summary: dict | None = None,
) -> AgentRunResult:
    """Run one evasion round (paper §4.3 aka S1) by delegating to ``agent_run``.

    Parameters
    ----------
    parent
        Seed variant for this round.  Forwarded as ``seed=`` to
        ``agent_run``.
    findings
        Paper findings that contextualize the attack.  Forwarded
        unchanged.
    repo
        Target repository slice.  Forwarded unchanged.
    agent_run
        The runtime callable that actually drives the LLM loop.
        Injected so unit tests can substitute a mock without
        monkey-patching module globals.
    run_config
        Runtime configuration.  ``memory_enabled`` and ``agent_strategy``
        are read here; the full config is forwarded to ``agent_run`` so
        downstream toggles still apply.
    round_index
        Zero-based index of the current evolution round.  Forwarded
        unchanged to ``agent_run`` (it flows into the LLM prompt).
    attack_log_path
        Optional path to the per-round attack log.  Forwarded unchanged.

    Returns
    -------
    AgentRunResult
        Whatever ``agent_run`` returns, verbatim.  No post-processing.
    """
    memory_enabled = run_config.memory_enabled

    # Forward whichever semantic level name the caller picked (paper §4.3
    # aka legacy "S1" defaulted to ``"evasion"``; the campaign driver now
    # may pass ``"free_mutation"`` for Phase-1 unconstrained discovery so
    # the agent can route through ``new_c.scripts``). Downstream consumers
    # (prompts, mutator, baseline adapter) accept the semantic vocabulary
    # as the canonical input.
    return agent_run(
        parent,
        findings,
        repo,
        search_space=search_space,
        memory_enabled=memory_enabled,
        run_config=run_config,
        round_index=round_index,
        attack_log_path=attack_log_path,
        last_round_summary=last_round_summary,
    )


__all__ = ["step_s1"]
