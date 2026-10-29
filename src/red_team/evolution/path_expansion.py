"""S2 free-expansion with mandatory 3-stage family enumeration.

Per paper §4.3 / Algorithm 1, once S1 bypasses we enter S2 *free rounds*
(capped by ``max_rounds``, default 3).  Each round is **not** simply
``agent_run(search_space="S2")``: it is a three-stage
``family_enumerate → agent_run → sandbox`` sequence that tests multiple
sibling techniques per round.

Flow for one round:

1. Derive ``current_technique`` from the parent (via
   ``metadata["technique_summary"]`` if present, else the objective's
   string value).
2. Call :func:`red_team.evolution.family_enumeration.family_enumerate`
   to produce up to ``top_k`` scored candidates.  An empty list aborts
   S2 altogether — we return whatever has been collected so far.
3. For each candidate, stash it on ``parent.metadata["family_hint"]``
   and run ``agent_run(search_space="S2", ablation={"family_hint": ...})``
   so the LLM sees the candidate as a hint (TODO: extend
   ``render_round_user`` to understand ``ablation.family_hint``
   natively).  Materialize the resulting variant, run it in the sandbox,
   and evaluate lethality via the injected ``oracle_lethal``.
4. Update ``parent`` to the best (bypassed + lethal) variant of the
   round if one exists; otherwise keep the round's parent so the next
   round still explores from the last confirmed success.

Ablation: if ``ablation["disable_family_enumeration"]`` is truthy we
skip brainstorm/filter and run exactly one ``agent_run(search_space="S2")``
per round — this is the "path expansion without family enumeration"
lesion.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.models import RepositorySlice
from core.schemas import SkillVariant
from red_team.campaign.run_config import RunConfig
from red_team.model_defaults import default_mutator_model
from red_team.evolution.family_enumeration import (
    FamilyCandidate,
    _iter_benign_skill_texts_cached,
    family_enumerate,
)
from sandbox.honeypot import HoneypotResult

__all__ = ["PathExpandResult", "path_expand"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PathExpandResult:
    """Outcome of a single candidate tested inside S2 path expansion.

    One :class:`PathExpandResult` is produced per (round, candidate)
    pair when the agent returns a mutation.  If the agent refused or the
    invariant check failed we skip the candidate instead of producing a
    result with ``mutation=None`` — downstream consumers should treat
    the returned list as "successfully materialized attempts".
    """

    mutation: SkillVariant
    tool_chain: list[dict[str, Any]]
    hp: HoneypotResult | None
    bypassed: bool | None
    lethal: bool
    family_candidate: dict[str, Any] | None
    score: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Shallow dict view for JSONL logging.

        :class:`SkillVariant` is serialized via its own ``to_dict`` so we
        don't accidentally mangle the ``(o, τ, κ)`` enum fields.
        """
        return {
            "mutation": self.mutation.to_dict(),
            "tool_chain": list(self.tool_chain),
            "hp": self.hp.to_dict() if self.hp is not None else None,
            "bypassed": self.bypassed,
            "lethal": self.lethal,
            "family_candidate": (
                dict(self.family_candidate)
                if self.family_candidate is not None
                else None
            ),
            "score": self.score,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fire_on_result(
    hook: Callable[["PathExpandResult"], None] | None,
    outcome: "PathExpandResult",
) -> None:
    """Per-entry checkpoint hook invocation, never raises.

    Lets the driver flush each path-expansion entry (audit + log +
    harvest + memory) as soon as it materializes, instead of waiting
    for the whole expansion batch.  A faulty hook must not break the
    inner loop, so we swallow + log any exception.
    """
    if hook is None:
        return
    try:
        hook(outcome)
    except Exception:  # pragma: no cover - defensive
        import logging
        logging.getLogger(__name__).exception("path_expand on_result hook crashed")


def _current_technique(parent: SkillVariant) -> str:
    """Derive a short technique label from the parent variant.

    We prefer ``metadata["technique_summary"]`` so the driver can hand us
    a round-specific description; absent that, fall back to the
    objective's string value so the brainstorm prompt is still grounded.
    """
    summary = parent.metadata.get("technique_summary")
    if isinstance(summary, str) and summary.strip():
        return summary
    return parent.o.value


def _variant_with_family_hint(
    parent: SkillVariant, candidate: FamilyCandidate | None
) -> SkillVariant:
    """Return a shallow clone of ``parent`` with ``family_hint`` metadata.

    The clone is necessary because :class:`SkillVariant` has ``slots=True``
    and we want to avoid mutating the caller's parent in place — the S2
    loop updates ``parent`` based on successful mutations, and sharing a
    metadata dict across rounds would leak stale hints.
    """
    new_metadata = dict(parent.metadata)
    if candidate is not None:
        new_metadata["family_hint"] = candidate.to_dict()
    else:
        new_metadata.pop("family_hint", None)
    return SkillVariant(
        o=parent.o,
        tau=parent.tau,
        c=copy.copy(parent.c) if parent.c else {},
        kappa=list(parent.kappa),
        d=parent.d,
        poison_index=parent.poison_index,
        chain=list(parent.chain),
        metadata=new_metadata,
    )


def _extract_tool_chain(agent_result: Any) -> list[dict[str, Any]]:
    """Pull the tool-call log off an :class:`AgentRunResult`.

    Baselines leave ``tool_chain`` as ``list[str]``; the agent runtime
    returns ``list[dict]``.  We coerce strings to single-key dicts so
    downstream consumers always see the same shape.
    """
    chain = getattr(agent_result, "tool_chain", None) or []
    coerced: list[dict[str, Any]] = []
    for item in chain:
        if isinstance(item, dict):
            coerced.append(item)
        else:
            coerced.append({"tool": str(item)})
    return coerced


def _run_one_candidate(
    parent: SkillVariant,
    candidate: FamilyCandidate | None,
    *,
    findings: list[Any],
    repo: RepositorySlice,
    agent_run: Callable[..., Any],
    run_in_sandbox: Callable[..., tuple[HoneypotResult, list[dict[str, Any]]]],
    oracle_lethal: Callable[..., bool],
    run_config: RunConfig,
    attack_log_path: Path | None,
    round_index: int,
    target_model: str,
    sandbox_container: str,
) -> PathExpandResult | None:
    """Execute one candidate end-to-end and return a :class:`PathExpandResult`.

    Returns ``None`` if the agent refused / hit an invariant violation
    (no mutation to execute) — the caller treats ``None`` as "skip".
    """
    import dataclasses as _dataclasses
    # Stuff the family candidate onto metadata so the prompt renderer can
    # see it.  Also build a RunConfig with the family_hint field set so
    # agent_run can access it via run_config.
    seed = _variant_with_family_hint(parent, candidate)
    if candidate is not None:
        run_config_with_hint = _dataclasses.replace(
            run_config, family_hint=candidate.to_dict() if candidate is not None else None
        )
    else:
        run_config_with_hint = run_config

    agent_result = agent_run(
        seed,
        findings,
        repo,
        search_space="path_expansion",  # paper §4.3 aka legacy "S2"
        round_index=round_index,
        run_config=run_config_with_hint,
        attack_log_path=attack_log_path,
    )

    mutation = getattr(agent_result, "mutation", None)
    if mutation is None:
        return None

    tool_chain = _extract_tool_chain(agent_result)
    # ``run_in_sandbox`` expects the skill-ID chain (list[str]), not the
    # agent's tool-call provenance. Use ``mutation.chain``.
    hp, sandbox_tool_calls = run_in_sandbox(
        mutation, list(mutation.chain), target_model, sandbox_container
    )
    # Merge sandbox-observed tool calls into the chain for full provenance.
    if sandbox_tool_calls:
        tool_chain = list(tool_chain) + list(sandbox_tool_calls)

    # The real bypass check is a blue-team re-audit owned by the
    # driver (campaign.runner._PerEntryHandler, fired as soon
    # as this PathExpandResult is appended).  Until that re-audit runs, we
    # surface ``bypassed=None`` ("unaudited") rather than the previous
    # always-True placeholder — None propagates through to log rows so a
    # missing audit is observably distinct from "audited and passed".
    bypassed: bool | None = None
    from red_team.oracle.sensor_metadata import with_skill_metadata
    reading = with_skill_metadata(hp, chain=list(mutation.chain or []))
    lethal = bool(oracle_lethal(mutation, reading, mutation.o))

    rationale = getattr(agent_result, "rationale", "") or ""
    family_dict = candidate.to_dict() if candidate is not None else None
    score = candidate.score if candidate is not None else 0.0

    return PathExpandResult(
        mutation=mutation,
        tool_chain=tool_chain,
        hp=hp,
        bypassed=bypassed,
        lethal=lethal,
        family_candidate=family_dict,
        score=score,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def path_expand(
    parent: SkillVariant,
    repo: RepositorySlice,
    *,
    agent_run: Callable[..., Any],
    run_in_sandbox: Callable[..., tuple[HoneypotResult, list[dict[str, Any]]]],
    oracle_lethal: Callable[..., bool],
    attack_log_path: Path | None,
    max_rounds: int = 3,
    top_k: int = 3,
    run_config: RunConfig | None = None,
    target_model: str = "glm-5",
    sandbox_container: str = "skillrt-openclaw-00",
    findings: list[Any] | None = None,
    mutator_model: str = default_mutator_model(),
    budget_usd: float = 5.0,
    on_result: Callable[["PathExpandResult"], None] | None = None,
) -> list[PathExpandResult]:
    """Run S2 path expansion with mandatory 3-stage family enumeration.

    Parameters mirror the task contract.  The function does **not** write
    to the attack log — the driver owns persistence.

    ``on_result`` (optional): driver hook called once per materialized
    :class:`PathExpandResult` immediately after it is appended.  Lets
    the runner audit + flush each entry to disk while the next
    candidate is still being brainstormed (per-entry checkpoint
    instead of end-of-expansion batch flush).  Exceptions are caught
    and logged so a faulty hook never aborts the loop.

    Returns a list of :class:`PathExpandResult`, one per successful
    materialization (the list may be empty if every round either failed
    the brainstorm/filter gate or the agent refused).
    """
    if run_config is None:
        run_config = RunConfig()
    findings = findings or []
    disable_family = not run_config.family_enumeration_enabled
    # C9: by default break after the first lethal hit each round so we
    # don't burn sandbox budget on the rest of the top-k once we have a
    # promotable mutation.  The "exhaustive" knob keeps the pre-C9
    # behaviour for the §"family-enum exhaustive" ablation column,
    # where we want recall data on every top-k candidate.
    exhaustive = run_config.path_expand_exhaustive

    # B6: accumulate every candidate name we have already brainstormed
    # so the next round's brainstorm prompt can ask the LLM to NOT
    # repeat them.  Belt-and-braces dedup also runs inside
    # ``family_enumerate`` for cases where the LLM ignores the ask.
    tried_names: set[str] = set()

    # C8: the SKILL.md tree doesn't change between rounds — read it
    # once and pass the cached texts into every brainstorm call.  The
    # ``_iter_benign_skill_texts_cached`` helper is itself memoised,
    # so this is mostly a hint for callers reading the code; the
    # actual disk I/O still gets deduped at the loader level.
    cached_skill_texts = _iter_benign_skill_texts_cached(None)

    # B5: pull the most recent blue-team findings off the parent so
    # brainstorm knows which patterns to avoid.  We accept either the
    # raw PaperFinding objects on ``parent.metadata['audit_findings']``
    # (set by the driver before calling path_expand) or the
    # ``findings`` arg already passed for agent_run context.  The
    # latter is the same list — driver passes it through both
    # channels — so de-dup by id() to avoid double counting.
    blue_findings: list[Any] = []
    metadata_findings = parent.metadata.get("audit_findings")
    if isinstance(metadata_findings, list):
        blue_findings.extend(metadata_findings)
    for f in findings or []:
        if not any(f is g for g in blue_findings):
            blue_findings.append(f)

    results: list[PathExpandResult] = []
    current_parent = parent

    for round_index in range(max_rounds):
        # ------------------------------------------------------------ ablation
        if disable_family:
            outcome = _run_one_candidate(
                current_parent,
                None,
                findings=findings,
                repo=repo,
                agent_run=agent_run,
                run_in_sandbox=run_in_sandbox,
                oracle_lethal=oracle_lethal,
                run_config=run_config,
                attack_log_path=attack_log_path,
                round_index=round_index,
                target_model=target_model,
                sandbox_container=sandbox_container,
            )
            if outcome is not None:
                results.append(outcome)
                _fire_on_result(on_result, outcome)
                # path_expand never has a real audit verdict — the driver
                # re-audits later — so advancement is gated on ``lethal``
                # only. This matches the previous behaviour exactly (the
                # old ``bypassed=True`` placeholder always short-circuited
                # the AND to just lethal); the change is only that we no
                # longer pretend to consult a verdict we do not have.
                if outcome.lethal:
                    current_parent = outcome.mutation
            continue

        # ------------------------------------------------------------ normal
        # B4/B5/B6/C8/E14: forward ablation hyperparams + the new
        # constraint inputs.  ``score_weights`` controls the additive
        # axis weights, ``blue_findings`` lets brainstorm see what the
        # blue team caught, ``tried_names`` prevents round-2/3 from
        # re-proposing round-1's siblings, ``cached_skill_texts``
        # avoids re-reading the SKILL.md tree on every round.
        candidates = family_enumerate(
            current_parent,
            _current_technique(current_parent),
            agent_run=agent_run,
            attack_log_path=attack_log_path,
            top_k=top_k,
            mutator_model=mutator_model,
            budget_usd=budget_usd,
            score_weights=run_config.family_score_weights,
            blue_findings=blue_findings,
            tried_names=tried_names,
            cached_skill_texts=cached_skill_texts,
            axis_diversity_strength=run_config.axis_diversity_strength,
        )
        if not candidates:
            # Brainstorm/filter failed this round — paper says "abort S2"
            # and return control to the driver.
            return results

        # B6: stash this round's candidate names so the next round's
        # prompt asks the LLM to skip them.  Done here (before sandbox
        # execution) so even abandoned candidates are remembered.
        tried_names.update(c.name for c in candidates if c.name)

        best_next: SkillVariant | None = None
        for candidate in candidates:
            outcome = _run_one_candidate(
                current_parent,
                candidate,
                findings=findings,
                repo=repo,
                agent_run=agent_run,
                run_in_sandbox=run_in_sandbox,
                oracle_lethal=oracle_lethal,
                run_config=run_config,
                attack_log_path=attack_log_path,
                round_index=round_index,
                target_model=target_model,
                sandbox_container=sandbox_container,
            )
            if outcome is None:
                continue
            results.append(outcome)
            _fire_on_result(on_result, outcome)
            # See ``disable_family`` branch above for why advancement is
            # gated on ``lethal`` alone here.  Candidates are pre-sorted
            # by score descending, so the first lethal hit is the highest
            # scoring lethal hit.
            if outcome.lethal and best_next is None:
                best_next = outcome.mutation
                # A2: stash the winning candidate's identity onto the
                # promoted mutation so the *next* round's
                # ``_current_technique`` derives the brainstorm seed from
                # what just succeeded, rather than reverting to the
                # parent's objective string. Without this update, rounds
                # 2 and 3 keep brainstorming siblings of the original
                # technique — the three "free rounds" collapse into
                # three near-identical samples instead of a chain.
                #
                # In-place mutation of the metadata dict is intentional:
                # the same SkillVariant becomes ``current_parent`` below
                # AND keeps the trace on the row already in ``results``,
                # which is useful for log analysis when reconstructing
                # which family hint each round tested.
                mechanism = (candidate.mechanism or "").strip()
                if candidate.name:
                    best_next.metadata["technique_summary"] = (
                        f"{candidate.name}: {mechanism}"
                        if mechanism else candidate.name
                    )
                # C9: by default we stop after the first lethal hit
                # this round — running the rest of top-k just to log
                # them is sandbox-budget-wasteful in production.  The
                # ``path_expand_exhaustive`` ablation flag preserves
                # the pre-C9 "run every candidate" behaviour for the
                # §"family-enum recall" ablation column where we
                # explicitly want stats on every top-k candidate.
                if not exhaustive:
                    break

        if best_next is not None:
            current_parent = best_next

    return results


