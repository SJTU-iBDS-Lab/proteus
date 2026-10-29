"""Phase-2 horizontal expansion driver — S2 (path) + S3 (surface).

Two entry points, both routing through :func:`_run_expansion`:

- :func:`run_phase2_expansion_only` — phase-2-only batch path.  Loads a
  frozen winner from ``run_config.start_variant_path``, runs S2/S3,
  writes its own minimal ``summary.json``.  Used when phase-1 has
  ALREADY finished for ALL seeds and the offline extractor has built
  shared ``memory.sqlite``.

- :func:`run_phase2_after_star` — fused-with-barrier path.  Called by
  :func:`run_evolution` when ``phase="both"`` AFTER its barrier hook
  has fired (all sibling seeds have finished phase-1 and the offline
  extractor has rebuilt shared memory).  Mutates the shared
  :class:`_RunState` so the runner's terminal summary reflects
  expansion telemetry.  Without a barrier this path is unsound — the
  runner enforces this by raising / degrading when no barrier is
  supplied.

The canonical Memory Phase-2 lifecycle (whichever shape the caller
chooses):

1. all seeds finish phase-1;
2. offline extractor builds shared ``memory.sqlite`` from harvest;
3. each winner runs S2/S3 against the frozen memory.

:func:`_run_expansion` swallows defensive exceptions and returns an
empty :class:`HorizontalExpansion` on crash.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from core.models import RepositorySlice
from core.schemas import PaperFinding, SkillVariant
from red_team.campaign import seams as _seams
from red_team.campaign.io import _RunPaths
from red_team.campaign.per_entry_handler import _PerEntryHandler
from red_team.campaign.run_config import RunConfig
from red_team.campaign.state import _RunState
from red_team.evolution.horizontal_expand import HorizontalExpansion
from red_team.log.memory.runtime_hook import MemoryRuntime

__all__ = [
    "_run_expansion",
    "run_phase2_after_star",
    "run_phase2_expansion_only",
]

_LOGGER = logging.getLogger(__name__)


def _run_expansion(
    s_prime: SkillVariant,
    repo: RepositorySlice,
    *,
    agent_run_fn: Callable[..., Any],
    sandbox_run_fn: Callable[..., Any] | None,
    oracle_fn: Callable[..., bool],
    horizontal_expand_fn: Callable[..., HorizontalExpansion],
    attack_log_path: Path,
    run_config: RunConfig,
    target_model: str,
    sandbox_container: str,
    mutator_model: str,
    free_expansions: int,
    surface_k: int,
    surface_k_prime: int,
    on_path_result: Callable[[Any], None] | None = None,
    on_surface_executed: Callable[[Any], None] | None = None,
) -> HorizontalExpansion:
    """Wrap ``horizontal_expand_fn`` with the shared sandbox seam and a
    defensive try/except.  Crashes degrade to an empty result so the
    caller can proceed to summary writing instead of bubbling up."""
    run_in_sandbox = (
        sandbox_run_fn if sandbox_run_fn is not None
        else (lambda v, c, tm, ct: _seams.sandbox_run(v, c, tm, ct, None)[:2])
    )
    try:
        return horizontal_expand_fn(
            s_prime, repo, agent_run=agent_run_fn,
            run_in_sandbox=run_in_sandbox, oracle_lethal=oracle_fn,
            attack_log_path=attack_log_path, run_config=run_config,
            target_model=target_model, sandbox_container=sandbox_container,
            mutator_model=mutator_model, path_max_rounds=free_expansions,
            surface_k=surface_k, surface_k_prime=surface_k_prime,
            on_path_result=on_path_result,
            on_surface_executed=on_surface_executed,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("horizontal_expand raised %s; proceeding to break", exc)
        return HorizontalExpansion()


def _recompute_funnel_succeeded(hx: HorizontalExpansion) -> None:
    """Per-entry callbacks set ``bypassed`` after expansion returns;
    ``funnel.succeeded`` was computed before that and is now stale.
    Recompute against the post-audit truth (Q4-A semantics)."""
    hx.surface_funnel.succeeded = sum(
        1 for r in hx.surface_funnel.executed
        if r.bypassed is True and r.lethal and r.post_hoc_ok
    )


def run_phase2_after_star(
    *,
    state: _RunState,
    paths: _RunPaths,
    run_config: RunConfig,
    s_prime: SkillVariant,
    repo: RepositorySlice,
    run_id: str,
    seed_id: str,
    target_model: str,
    mutator_model: str,
    free_expansions: int,
    surface_k: int,
    surface_k_prime: int,
    blue_team: str,
    blue_team_fn: Callable[..., list[PaperFinding]] | None,
    sandbox_container: str,
    backend_tag: str,
    agent_run_fn: Callable[..., Any],
    oracle_fn: Callable[..., bool],
    horizontal_expand_fn: Callable[..., HorizontalExpansion],
    sandbox_run_fn: Callable[..., Any] | None,
    memory_runtime: MemoryRuntime,
) -> None:
    """Phase-2 inline after a barrier-released phase-1 ★.

    Caller (``run_evolution``) must invoke this only after the phase-2
    barrier has fired (i.e. all sibling seeds have completed phase-1
    and the offline extractor has rebuilt shared memory).  Mutates
    ``state`` (expansion_audits, expansion_audit_cache_hits) so the
    runner's terminal summary reflects expansion telemetry.
    """
    t0 = time.monotonic()
    _LOGGER.info(
        "expansion.start  parent=%s  free=%d  surface_k=%d  surface_k'=%d",
        s_prime.content_hash()[:12],
        free_expansions, surface_k, surface_k_prime,
    )
    handler = _PerEntryHandler(
        repo=repo, blue_team=blue_team, blue_team_fn=blue_team_fn,
        sandbox_container=sandbox_container,
        log_path=paths.log, harvest_path=paths.harvest,
        backend_tag=backend_tag,
        run_id=run_id, seed_id=seed_id, parent_variant=s_prime,
        target_model=target_model, memory_runtime=memory_runtime,
    )
    hx = _run_expansion(
        s_prime, repo, agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_run_fn, oracle_fn=oracle_fn,
        horizontal_expand_fn=horizontal_expand_fn,
        attack_log_path=paths.attack_log, run_config=run_config,
        target_model=target_model, sandbox_container=sandbox_container,
        mutator_model=mutator_model, free_expansions=free_expansions,
        surface_k=surface_k, surface_k_prime=surface_k_prime,
        on_path_result=handler.on_path_result,
        on_surface_executed=handler.on_surface_executed,
    )
    _recompute_funnel_succeeded(hx)
    audits, hits = handler.finalize()
    state.expansion_audits += audits
    state.expansion_audit_cache_hits += hits
    _LOGGER.info(
        "expansion.done   path=%d  surface_executed=%d  audits=%d  "
        "cache_hits=%d  elapsed=%.1fs",
        len(hx.path_results), len(hx.surface_funnel.executed),
        audits, hits, time.monotonic() - t0,
    )


def run_phase2_expansion_only(
    *,
    paths: _RunPaths,
    run_config: RunConfig,
    seed: SkillVariant,
    repo: RepositorySlice,
    run_id: str,
    target_model: str,
    mutator_model: str,
    free_expansions: int,
    surface_k: int,
    surface_k_prime: int,
    blue_team: str,
    blue_team_fn: Callable[..., list[PaperFinding]] | None,
    sandbox_container: str,
    agent_run_fn: Callable[..., Any],
    oracle_fn: Callable[..., bool],
    horizontal_expand_fn: Callable[..., HorizontalExpansion],
    sandbox_run_fn: Callable[..., Any] | None,
    memory_runtime: MemoryRuntime,
) -> Path:
    """Phase-2-only driver — load a phase-1 winner and run S2/S3.

    Falls back to ``seed`` when ``start_variant_path`` is missing
    (degenerate but won't crash).  Writes its own minimal summary.
    """
    seed_id = str(seed.metadata.get("seed_id") or seed.content_hash())
    winner = seed
    sv_path = run_config.start_variant_path
    if sv_path:
        try:
            payload = json.loads(Path(sv_path).read_text(encoding="utf-8"))
            winner = SkillVariant.from_dict(payload["winner"])
            _LOGGER.info(
                "expansion_only loaded start variant from %s (hash=%s)",
                sv_path, payload.get("winner_hash", "?")[:16],
            )
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning(
                "expansion_only failed to load start_variant_path=%s: %s; "
                "falling back to seed", sv_path, exc,
            )
    _LOGGER.info(
        "expansion_only.start  free=%d  surface_k=%d  surface_k'=%d",
        free_expansions, surface_k, surface_k_prime,
    )
    handler = _PerEntryHandler(
        repo=repo, blue_team=blue_team, blue_team_fn=blue_team_fn,
        sandbox_container=sandbox_container,
        log_path=paths.log, harvest_path=paths.harvest,
        backend_tag="expansion_only",
        run_id=run_id, seed_id=seed_id, parent_variant=winner,
        target_model=target_model, memory_runtime=memory_runtime,
    )
    hx = _run_expansion(
        winner, repo, agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_run_fn, oracle_fn=oracle_fn,
        horizontal_expand_fn=horizontal_expand_fn,
        attack_log_path=paths.attack_log, run_config=run_config,
        target_model=target_model, sandbox_container=sandbox_container,
        mutator_model=mutator_model, free_expansions=free_expansions,
        surface_k=surface_k, surface_k_prime=surface_k_prime,
        on_path_result=handler.on_path_result,
        on_surface_executed=handler.on_surface_executed,
    )
    _recompute_funnel_succeeded(hx)
    audits, hits = handler.finalize()
    _LOGGER.info(
        "expansion_only.audit.done  audits=%d  cache_hits=%d "
        "path_results=%d  surface_executed=%d",
        audits, hits,
        len(hx.path_results), len(hx.surface_funnel.executed),
    )
    summary = {
        "run_id": run_id,
        "seed_id": seed_id,
        "phase": "expansion_only",
        "target_model": target_model,
        "winner_hash": winner.content_hash(),
        "path_results": len(hx.path_results),
        "surface_executed": len(hx.surface_funnel.executed),
        "audits_issued": audits,
    }
    paths.summary.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return paths.run_dir
