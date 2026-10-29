"""Campaign runner — thin per-seed entry point.

``run_evolution`` is the public API for **one seed × one or two phases**.
Phase routing:

- ``"main_only"`` — :func:`phase1_main.run_phase1` only.  Dumps
  ``phase1_winner.json`` on exit.
- ``"expansion_only"`` — :func:`phase2_expand.run_phase2_expansion_only`
  only.  Loads a winner from ``run_config.start_variant_path``.
- ``"both"`` — phase-1 → **barrier wait** → phase-2 inline.  Real
  two-phase mode: phase-1 runs, the winner is dumped, then the caller-
  supplied ``phase2_barrier`` callable blocks until all sibling seeds
  have also finished phase-1 (and the offline extractor has rebuilt
  shared memory).  After the barrier returns (optionally yielding a
  refreshed :class:`MemoryRuntime`) phase-2 runs against the post-
  barrier memory.  Without a ``phase2_barrier`` this mode degrades to
  ``main_only`` with a warning — there is no sound way to run phase-2
  before sibling seeds finish phase-1.

Why the barrier matters: inlining phase-2 inside ``run_evolution``
without waiting would mean seed-A's phase-2 (which retrieves from
``memory.sqlite``) sees memory that does NOT yet contain seed-B's
phase-1 contributions.  The Memory Phase-2 plan mandates "memory
frozen between phases"; the barrier is what enforces that.

Heavy work lives in the phase modules; this file owns only:

- public ``run_evolution`` signature (frozen for callers);
- path resolution + run-banner / file-handler attach;
- the memory-runtime context that deep callers reach via
  :func:`current_memory_runtime`;
- phase dispatch + post-phase-1 winner dump + barrier coordination.

Re-exports for tests: :class:`_PerEntryHandler`, :class:`_RunState`,
:func:`_bypass_from_findings`, ``_seams``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from core.models import RepositorySlice
from core.schemas import PaperFinding, SkillVariant
from red_team.campaign import phase1_main as _phase1
from red_team.campaign import phase2_expand as _phase2
from red_team.campaign import seams as _seams  # re-exported for tests
from red_team.model_defaults import default_mutator_model
from red_team.campaign.io import (
    _default_run_id,
    _dump_phase1_winner,
    _resolve_paths,
)
from red_team.campaign.per_entry_handler import _PerEntryHandler  # re-exported
from red_team.campaign.run_config import RunConfig
from red_team.campaign.state import _bypass_from_findings, _RunState  # re-exported
from red_team.evolution.horizontal_expand import HorizontalExpansion
from red_team.log.memory.runtime_context import current_memory_runtime
from red_team.log.memory.runtime_hook import MemoryRuntime, create_memory_runtime
from red_team.util.run_logging import attach_run_file_handler, emit_run_banner
from sandbox.honeypot import HoneypotResult

# A phase-2 barrier callable.  ``run_evolution`` invokes this AFTER
# phase-1 completes (and the winner has been dumped) and BEFORE phase-2
# starts.  The callable should block until all sibling seeds in the
# batch have also finished phase-1 and the offline extractor has
# rebuilt shared memory.  It may return a fresh :class:`MemoryRuntime`
# pointing at the post-extractor shared memory; if it returns ``None``
# the existing per-run memory_runtime is reused (only useful when the
# caller has already updated the underlying sqlite in place).
Phase2BarrierFn = Callable[[Path], "MemoryRuntime | None"]

__all__ = [
    "run_evolution",
    # Re-exported for tests reaching into runner internals:
    "_PerEntryHandler",
    "_RunState",
    "_bypass_from_findings",
    "_seams",
]

_LOGGER = logging.getLogger(__name__)


def run_evolution(
    seed: SkillVariant,
    repo: RepositorySlice,
    *,
    target_model: str = "glm-5",
    mutator_model: str = default_mutator_model(),
    rounds: int = 20,
    free_expansions: int = 3,
    surface_k: int = 10,
    surface_k_prime: int = 3,
    ablation: dict[str, Any] | None = None,
    blue_team: str = "ours",
    sandbox_container: str = "skillrt-openclaw-00",
    run_id: str | None = None,
    output_root: Path | str = "outputs/evolution",
    blue_team_fn: Callable[..., list[PaperFinding]] | None = None,
    sandbox_run_fn: Callable[..., tuple[HoneypotResult, list[dict[str, Any]]]] | None = None,
    agent_run_fn: Callable[..., Any] | None = None,
    oracle_fn: Callable[..., bool] | None = None,
    horizontal_expand_fn: Callable[..., HorizontalExpansion] | None = None,
    phase2_barrier: Phase2BarrierFn | None = None,
) -> Path:
    """Drive Algorithm 1 for ``rounds`` rounds (with optional phase-2 break).

    ``phase2_barrier`` is required when ``ablation['phase'] == 'both'``.
    It is invoked after phase-1 winner dump and must block until all
    sibling seeds have completed phase-1 (and the offline extractor has
    rebuilt shared memory).  It may return a fresh ``MemoryRuntime``
    bound to the post-extractor memory; if ``None`` is returned the
    per-run memory_runtime is reused.
    """
    run_config = RunConfig.from_ablation(dict(ablation or {}))
    agent_run_fn = agent_run_fn or _seams.default_agent_run
    # Pre-bind mutator_model onto agent_run_fn so the CLI arg
    # (--mutator-model) actually reaches agent_run() instead of being
    # silently dropped by step_s1 which never forwards it.  Discovered
    # 2026-05-02 while smoke-testing kimi-k2.6 — mutator.prompt log
    # showed model=deepseek-v4-pro despite --mutator-model kimi-k2.6.
    import functools as _ft
    agent_run_fn = _ft.partial(agent_run_fn, mutator_model=mutator_model)
    oracle_fn = oracle_fn or _seams.default_oracle_lethal
    horizontal_expand_fn = horizontal_expand_fn or _seams.default_horizontal_expand

    run_id = run_id or _default_run_id(seed, repo, target_model)
    paths = _resolve_paths(output_root, run_id)
    state = _RunState(current_seed=seed)
    seed_id = str(seed.metadata.get("seed_id") or seed.content_hash())
    search_space = run_config.search_space
    phase = run_config.phase

    log_cm = attach_run_file_handler(paths.run_dir)
    log_cm.__enter__()
    try:
        emit_run_banner(
            _LOGGER,
            run_id=run_id, seed_id=seed_id,
            repo_id=repo.repo_id or "", target_model=target_model,
            mutator_model=mutator_model, blue_team=blue_team,
            rounds=rounds, out_dir=paths.run_dir,
        )
        memory_runtime = create_memory_runtime(
            memory_db=paths.memory_db, run_config=run_config,
            seed_id=seed_id, target_model=target_model,
        )
        with current_memory_runtime(memory_runtime):
            if phase == "expansion_only":
                return _phase2.run_phase2_expansion_only(
                    paths=paths, run_config=run_config,
                    seed=seed, repo=repo, run_id=run_id,
                    target_model=target_model, mutator_model=mutator_model,
                    free_expansions=free_expansions, surface_k=surface_k,
                    surface_k_prime=surface_k_prime,
                    blue_team=blue_team, blue_team_fn=blue_team_fn,
                    sandbox_container=sandbox_container,
                    agent_run_fn=agent_run_fn, oracle_fn=oracle_fn,
                    horizontal_expand_fn=horizontal_expand_fn,
                    sandbox_run_fn=sandbox_run_fn,
                    memory_runtime=memory_runtime,
                )

            run_dir = _phase1.run_phase1(
                paths=paths, state=state, run_config=run_config,
                search_space=search_space, rounds=rounds,
                blue_team=blue_team, blue_team_fn=blue_team_fn,
                sandbox_container=sandbox_container,
                target_model=target_model, mutator_model=mutator_model,
                agent_run_fn=agent_run_fn, oracle_fn=oracle_fn,
                sandbox_run_fn=sandbox_run_fn,
                seed=seed, repo=repo, run_id=run_id,
                memory_runtime=memory_runtime,
            )
            winner_path = _dump_phase1_winner(
                paths, state.current_seed, seed_id, target_model
            )

            if phase != "both":
                return run_dir

            # ``phase=="both"``: real two-phase mode.  Block on the
            # caller's barrier until all sibling seeds have finished
            # phase-1 (and the offline extractor has rebuilt memory).
            if phase2_barrier is None:
                _LOGGER.warning(
                    "phase='both' requires a phase2_barrier; without one "
                    "phase-2 would race ahead of sibling seeds' phase-1. "
                    "Falling back to main_only behaviour for this run."
                )
                return run_dir

            if state.star_round is None:
                # No ★ → nothing to expand.  Still call the barrier so
                # the batch coordinator's count stays consistent.
                phase2_barrier(winner_path)
                return run_dir

            if not _is_expansion_enabled(run_config):
                # A5 ablation: expansion explicitly disabled.  Honour
                # the barrier (count stays consistent) but skip phase-2.
                phase2_barrier(winner_path)
                return run_dir

            refreshed = phase2_barrier(winner_path)
            if refreshed is not None:
                memory_runtime = refreshed

            _phase2.run_phase2_after_star(
                state=state, paths=paths, run_config=run_config,
                s_prime=state.current_seed, repo=repo,
                run_id=run_id, seed_id=seed_id,
                target_model=target_model, mutator_model=mutator_model,
                free_expansions=free_expansions, surface_k=surface_k,
                surface_k_prime=surface_k_prime,
                blue_team=blue_team, blue_team_fn=blue_team_fn,
                sandbox_container=sandbox_container,
                backend_tag="agent",
                agent_run_fn=agent_run_fn, oracle_fn=oracle_fn,
                horizontal_expand_fn=horizontal_expand_fn,
                sandbox_run_fn=sandbox_run_fn,
                memory_runtime=memory_runtime,
            )
            return run_dir
    finally:
        log_cm.__exit__(None, None, None)


def _is_expansion_enabled(run_config: RunConfig) -> bool:
    """Lazy import — avoids a top-level cycle through ablations."""
    from red_team.campaign import ablations as _ablations
    return _ablations.is_expansion_enabled(run_config)
