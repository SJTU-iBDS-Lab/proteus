"""Horizontal expansion — S2 (path) + S3 (surface) merge.

Per paper §4.3 / Algorithm 1 line 12, once a variant has bypassed at S1 we
run **horizontal expansion**: first S2 free rounds with mandatory 3-stage
family enumeration, then S3 cross-objective surface ideation seeded from the
*best* S2 outcome.  This module is the thin orchestrator that wires
:func:`red_team.evolution.path_expansion.path_expand` and
:func:`red_team.evolution.surface_expansion.surface_expand` together,
implementing the "pick best S2 variant as S3 input" rule.

Selection policy for the S3 seed:

1. Prefer the :class:`PathExpandResult` with ``bypassed=True and lethal=True``
   and the highest ``score`` (family-enumeration ``benign_cover`` — the paper's
   plausibility heuristic).  Ties are broken by first-seen order (stable).
2. If no S2 row satisfies both gates — including when ``path_results`` is
   empty — fall back to the original ``parent`` so S3 still gets a chance to
   find a cross-objective motif.

Ablation: ``ablation.disable_expansion=True`` short-circuits to an empty
:class:`HorizontalExpansion` without calling ``path_expand`` or
``surface_expand`` — this is the "no horizontal expansion" lesion.
The flag is **not** forwarded into either sub-call after the
short-circuit check: the sub-modules have their own ``disable_expansion``
semantics (``surface_expand`` bails out identically; ``path_expand`` ignores
it), so forwarding preserves invariants without double-handling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.models import RepositorySlice
from core.schemas import SkillVariant
from red_team.campaign.run_config import RunConfig
from red_team.model_defaults import default_mutator_model
from red_team.evolution.path_expansion import PathExpandResult, path_expand
from red_team.evolution.surface_expansion import (
    SurfaceFunnel,
    surface_expand,
)
from sandbox.honeypot import HoneypotResult

__all__ = ["HorizontalExpansion", "horizontal_expand"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HorizontalExpansion:
    """Aggregated S2 + S3 outcome for a single horizontal-expansion call.

    ``path_results`` is the raw :class:`PathExpandResult` list returned by
    :func:`path_expand` (may be empty if every round failed the
    brainstorm/filter gate or the agent refused).  ``surface_funnel`` is the
    :class:`SurfaceFunnel` returned by :func:`surface_expand` (also possibly
    empty if ideation failed or S3 was skipped).
    """

    path_results: list[PathExpandResult] = field(default_factory=list)
    surface_funnel: SurfaceFunnel = field(default_factory=SurfaceFunnel)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_best_mutation(
    path_results: list[PathExpandResult],
) -> SkillVariant | None:
    """Return the mutation from the best lethal row, preferring confirmed bypass.

    Tri-state ordering on ``bypassed`` (which can now be ``None`` when
    the row has not been audited yet — see
    :func:`red_team.evolution.path_expansion.path_expand`):

    1. **Confirmed**: ``bypassed is True and lethal``.  Highest-scoring
       wins; ties broken by first-seen order.
    2. **Pending audit**: ``bypassed is None and lethal``.  Used only
       when no confirmed row exists.  ``horizontal_expand`` runs the
       per-entry audit hook (``_PerEntryHandler``) *as each row
       is appended*, so by the time we reach ``_pick_best_mutation`` for
       S3 some rows may already have ``bypassed=True``; pre-hook tests
       that supplied no callback see ``bypassed=None`` here.
    3. **Rejected**: anything else (``bypassed is False``, or
       ``lethal=False``).  Skipped.

    Returns ``None`` when both pickable buckets are empty so the caller
    falls back to the parent variant.
    """
    confirmed: list[PathExpandResult] = []
    pending: list[PathExpandResult] = []
    for row in path_results:
        if not row.lethal:
            continue
        if row.bypassed is True:
            confirmed.append(row)
        elif row.bypassed is None:
            pending.append(row)
        # ``bypassed is False`` => audited and rejected; skip.

    pool = confirmed or pending
    best: PathExpandResult | None = None
    for row in pool:
        if best is None or row.score > best.score:
            best = row
    return best.mutation if best is not None else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def horizontal_expand(
    parent: SkillVariant,
    repo: RepositorySlice,
    *,
    agent_run: Callable[..., Any],
    run_in_sandbox: Callable[..., tuple[HoneypotResult, list[dict[str, Any]]]],
    oracle_lethal: Callable[..., bool],
    attack_log_path: Path | None,
    run_config: RunConfig | None = None,
    target_model: str = "glm-5",
    sandbox_container: str = "skillrt-openclaw-00",
    mutator_model: str = default_mutator_model(),
    path_max_rounds: int = 3,
    path_top_k: int = 3,
    surface_k: int = 10,
    surface_k_prime: int = 3,
    on_path_result: Callable[[PathExpandResult], None] | None = None,
    on_surface_executed: Callable[[Any], None] | None = None,
) -> HorizontalExpansion:
    """Run S2 path expansion then S3 surface expansion on the best S2 variant.

    Parameters mirror the task contract.  The function does **not** write to
    the attack log — the driver owns persistence.

    ``run_config.expansion_enabled=False`` returns an empty
    :class:`HorizontalExpansion` without invoking either sub-module; any
    other run_config knob is forwarded verbatim so the sub-modules' own
    semantics remain intact.

    ``on_path_result`` / ``on_surface_executed`` (optional): per-entry
    checkpoint hooks forwarded to :func:`path_expand` and
    :func:`surface_expand`.  Each is invoked once per materialized
    entry so the driver can audit + flush incrementally rather than
    waiting for the whole expansion to complete.
    """
    if run_config is None:
        run_config = RunConfig()

    # -- expansion short-circuit -----------------------------------------------
    if not run_config.expansion_enabled:
        return HorizontalExpansion()

    # -- S2 path expansion -----------------------------------------------------
    path_results = path_expand(
        parent,
        repo,
        agent_run=agent_run,
        run_in_sandbox=run_in_sandbox,
        oracle_lethal=oracle_lethal,
        attack_log_path=attack_log_path,
        max_rounds=path_max_rounds,
        top_k=path_top_k,
        run_config=run_config,
        target_model=target_model,
        sandbox_container=sandbox_container,
        on_result=on_path_result,
    )

    # -- pick best S2 variant or fall back to parent ---------------------------
    best_mutation = _pick_best_mutation(path_results)
    s3_seed = best_mutation if best_mutation is not None else parent

    # -- S3 surface expansion --------------------------------------------------
    surface_funnel = surface_expand(
        s3_seed,
        repo,
        agent_run=agent_run,
        run_in_sandbox=run_in_sandbox,
        oracle_lethal=oracle_lethal,
        attack_log_path=attack_log_path,
        k=surface_k,
        k_prime=surface_k_prime,
        run_config=run_config,
        target_model=target_model,
        sandbox_container=sandbox_container,
        mutator_model=mutator_model,
        on_executed=on_surface_executed,
    )

    return HorizontalExpansion(
        path_results=path_results,
        surface_funnel=surface_funnel,
    )
