"""Phase-1 main loop — Algorithm 1's evasion stage (audit → mutate → sandbox).

One round of phase-1 corresponds to:

1. blue-team audit on the current seed;
2. apply ablation feedback hooks + inject prior round's runtime feedback;
3. dispatch the mutator (LLM agent or ablation random);
4. early-exit when the mutator produced no mutation (refusal / budget);
5. mutation invariant check (κ-edit forbiddance, etc.);
6. apply sandbox rewriters + sandbox the variant;
7. oracle lethality + synthesise next-round runtime feedback;
8. write log row + attack-log entry + harvest row + memory hook;
9. ★ check (bypass ∧ lethal); update ``state.current_seed`` to the
   winning ``s_prime`` whether or not ★ fires (so the runner can read
   the canonical winner via :class:`_RunState` after the loop ends).

This module **does not** trigger phase-2 expansion.  When ★ fires,
``_run_one_round`` records ``state.star_round`` and returns ``True`` so
the loop breaks; the runner then optionally dispatches phase-2 in a
separate step (``phase="both"``) or leaves the run as phase-1-only
(``phase="main_only"``).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from core.models import RepositorySlice
from core.schemas import PaperFinding, SkillVariant
from red_team.baselines.blackbox_rewrite import BlackboxRewrite
from red_team.baselines.llm_zeroshot import LLMZeroShot
from red_team.baselines.random_mutation import RandomMutation
from red_team.campaign import ablations as _ablations
from red_team.campaign import io as _io
from red_team.campaign import seams as _seams
from red_team.campaign.io import _RunPaths
from red_team.campaign.run_config import RunConfig
from red_team.campaign.state import (
    _bypass_from_findings,
    _RunState,
    build_last_round_summary,
)
from red_team.evolution.evasion import step_s1
from red_team.evolution.search_space import InvariantViolation
from red_team.feedback.runtime_feedback import synthesize_runtime_feedback
from red_team.log.harvest_writer import (
    append_harvest_row,
    build_main_round_harvest_row,
)
from red_team.log.memory.runtime_hook import MemoryRuntime, create_memory_runtime
from red_team.util.run_logging import emit_run_complete
from sandbox.target_reply import extract_target_reply

if TYPE_CHECKING:
    from red_team.external_baselines.trace import VictimTrace

__all__ = [
    "_log_abort",
    "_run_one_round",
    "_run_single_round",
    "run_phase1",
]

_LOGGER = logging.getLogger(__name__)


def _run_single_round(
    *,
    seed: SkillVariant,
    findings_for_agent: list[PaperFinding],
    repo: RepositorySlice,
    search_space: str,
    run_config: RunConfig,
    round_index: int,
    attack_log_path: Path,
    agent_run_fn: Callable[..., Any],
    last_round_summary: dict | None = None,
    victim_observation: "VictimTrace | None" = None,
) -> tuple[Any, str]:
    """Dispatch to either the S1 agent loop or an ablation short-circuit."""
    short = _ablations.select_agent_dispatch(run_config)
    if short is not None:
        _, backend_tag = short
        if backend_tag == "random_baseline":
            ar = RandomMutation().agent_run(
                seed, findings_for_agent, repo, search_space,
            )
        elif backend_tag == "llm_zeroshot":
            ar = LLMZeroShot().agent_run(
                seed, findings_for_agent, repo, search_space,
            )
        elif backend_tag == "blackbox_rewrite":
            ar = BlackboxRewrite().agent_run(
                seed, findings_for_agent, repo, search_space,
                round_index=round_index,
            )
        elif backend_tag == "pyrit_standalone":
            # Lazy import keeps the optional PyRIT dependency out of all
            # other campaign configurations.
            from red_team.baselines.pyrit_baseline import PyRITStandalone
            ar = PyRITStandalone().agent_run(
                seed, findings_for_agent, repo, search_space,
            )
        elif backend_tag == "skill_inject_bo5":
            from red_team.external_baselines.campaign import SkillInjectCampaign
            ar = SkillInjectCampaign().agent_run(
                seed, findings_for_agent, repo, search_space,
            )
        elif backend_tag == "skillject_trace":
            from red_team.external_baselines.campaign import SkillJectCampaign
            ar = SkillJectCampaign().agent_run(
                seed, findings_for_agent, repo, search_space,
                observation=victim_observation,
            )
        else:
            raise ValueError(
                f"unknown programmatic baseline tag: {backend_tag!r}"
            )
        if ar.stop_reason != "early_stop":
            ar.stop_reason = "programmatic_fallback"
        return ar, backend_tag
    ar = step_s1(
        seed, findings_for_agent, repo,
        agent_run=agent_run_fn, run_config=run_config,
        round_index=round_index, attack_log_path=attack_log_path,
        search_space=search_space,
        last_round_summary=last_round_summary,
    )
    return ar, "agent"


def _log_abort(
    paths: _RunPaths,
    state: _RunState,
    seed: SkillVariant,
    ar: Any,
    backend_tag: str,
    stop_reason: str,
    rationale: str,
    round_index: int,
    audit_findings: list[PaperFinding],
    phase: str = "free_mutation",
) -> None:
    """Emit log+attack-log pair for a round that never reached the sandbox."""
    _io.write_log_row(paths.log, _io.build_log_row(
        round_index=round_index, variant=state.current_seed,
        findings=audit_findings, hp=None, bypassed=False, lethal_flag=False,
        backend=backend_tag, stop_reason=stop_reason, rationale=rationale,
        tool_chain=list(getattr(ar, "tool_chain", []) or []),
        cost_usd=float(getattr(ar, "cost_usd", 0.0) or 0.0),
        is_expansion=False, phase=phase,
    ))
    _io.append_attack_log(
        paths.attack_log, round_index=round_index, seed=seed, variant=None,
        bypassed=False, lethal_flag=False, rationale=rationale,
        tool_chain=list(getattr(ar, "tool_chain", []) or []), is_expansion=False,
    )


def _run_one_round(
    t: int,
    *,
    state: _RunState,
    paths: _RunPaths,
    run_config: RunConfig,
    search_space: str,
    rounds: int,
    blue_team: str,
    blue_team_fn: Callable[..., list[PaperFinding]] | None,
    sandbox_container: str,
    target_model: str,
    agent_run_fn: Callable[..., Any],
    oracle_fn: Callable[..., bool],
    sandbox_run_fn: Callable[..., Any] | None,
    seed: SkillVariant,
    repo: RepositorySlice,
    run_id: str,
    seed_id: str,
    memory_runtime: MemoryRuntime,
) -> bool:
    """Execute one round of Algorithm 1.

    Mutates ``state`` in place.  Returns ``True`` when ★ (bypass+lethal)
    fires (caller should break the loop), ``False`` otherwise.

    Both early-exit cases (no mutation produced, invariant violation)
    return ``False`` and leave ``state.current_seed`` unchanged so the
    next round retries against the same parent.
    """
    _LOGGER.info(
        "[r=%d/%d] START  phase=%s  seed=%s",
        t, rounds, search_space,
        (state.current_seed.metadata or {}).get("seed_id")
        or state.current_seed.content_hash()[:12],
    )

    gated = run_config.audit_gate == "lethal"
    if gated:
        carried_findings = list(state.last_audit_findings)
        state.last_audit_findings = []
        audit_findings: list[PaperFinding] = []
        audit_bypass = False
    else:
        audit_t0 = time.monotonic()
        _LOGGER.info(
            "[r=%d/%d] audit.start  blue=%s  chain=%s",
            t, rounds, blue_team, list(state.current_seed.chain or []),
        )
        audit_findings = _seams.call_blue_team(
            state.current_seed, repo, blue_team, blue_team_fn,
            sandbox_container=sandbox_container,
        )
        audit_dt = time.monotonic() - audit_t0
        audit_bypass = _bypass_from_findings(audit_findings)
        carried_findings = list(audit_findings)
        _LOGGER.info(
            "[r=%d/%d] audit.done   verdict=%s  findings=%d  elapsed=%.1fs",
            t, rounds,
            "APPROVE" if audit_bypass else "REJECT",
            len(audit_findings or []), audit_dt,
        )

    findings_for_agent = _ablations.apply_feedback_hooks(
        carried_findings, run_config
    )
    # Inject prior round's runtime-layer feedback so the mutator can
    # reason about runtime executability (gray-box signal: we report
    # observed sandbox state, we do NOT tell the mutator which non-exec
    # tool to use as a workaround).
    if state.last_runtime_findings:
        findings_for_agent = list(findings_for_agent) + list(
            state.last_runtime_findings
        )
        state.last_runtime_findings = []

    mut_t0 = time.monotonic()
    _LOGGER.info(
        "[r=%d/%d] mutator.start  space=%s  hints=%d",
        t, rounds, search_space, len(findings_for_agent or []),
    )
    ar, backend_tag = _run_single_round(
        seed=state.current_seed, findings_for_agent=findings_for_agent,
        repo=repo, search_space=search_space, run_config=run_config,
        round_index=t, attack_log_path=paths.attack_log,
        agent_run_fn=agent_run_fn,
        last_round_summary=state.last_round_summary,
        victim_observation=state.last_victim_trace,
    )
    mut_dt = time.monotonic() - mut_t0
    state.total_cost += float(getattr(ar, "cost_usd", 0.0) or 0.0)
    _LOGGER.info(
        "[r=%d/%d] mutator.done   stop=%s  cost=$%.4f  elapsed=%.1fs",
        t, rounds,
        str(getattr(ar, "stop_reason", "") or ""),
        float(getattr(ar, "cost_usd", 0.0) or 0.0),
        mut_dt,
    )

    if ar.mutation is None:
        _LOGGER.warning(
            "[r=%d/%d] mutator produced no mutation (stop=%s); aborting round",
            t, rounds, str(getattr(ar, "stop_reason", "") or "refusal"),
        )
        _log_abort(
            paths, state, seed, ar, backend_tag,
            stop_reason=str(getattr(ar, "stop_reason", "") or "refusal"),
            rationale=str(getattr(ar, "rationale", "") or ""),
            round_index=t, audit_findings=audit_findings, phase=search_space,
        )
        # Surface the abort to the next round's prompt so the mutator
        # knows it ran out of tool calls / refused last time.
        state.last_round_summary = build_last_round_summary(
            round_index=t, parent_chain=list(state.current_seed.chain or []),
            audit_findings=[], bypassed=None if gated else audit_bypass,
            s_prime=None, lethal_flag=False,
            stop_reason=str(getattr(ar, "stop_reason", "") or "refusal"),
        )
        if str(getattr(ar, "stop_reason", "") or "") == "early_stop":
            state.strategy_early_stop = True
        return False

    s_prime = ar.mutation
    try:
        _ablations.check_mutation_invariants(state.current_seed, s_prime, run_config)
    except InvariantViolation as exc:
        _LOGGER.warning(
            "[r=%d/%d] invariant violation: %s; aborting round",
            t, rounds, exc,
        )
        _log_abort(
            paths, state, seed, ar, backend_tag,
            stop_reason="invariant_violation_a8", rationale=str(exc),
            round_index=t, audit_findings=audit_findings, phase=search_space,
        )
        state.last_round_summary = build_last_round_summary(
            round_index=t, parent_chain=list(state.current_seed.chain or []),
            audit_findings=[], bypassed=None if gated else audit_bypass,
            s_prime=None, lethal_flag=False,
            stop_reason="invariant_violation_a8",
        )
        return False

    s_prime_box, forced, chain_orig, chain_used = (
        _ablations.apply_sandbox_rewriters(s_prime, run_config)
    )
    if forced:
        # The rewritten artifact is the evaluated variant end-to-end:
        # sandbox, oracle, audit, logs, and the next parent see identical bytes.
        s_prime = s_prime_box
    _io.persist_mutation_md(paths.mutations_dir, t, s_prime, _seams.DEFAULT_SKILL_ROOT)

    sb_t0 = time.monotonic()
    sb_t0_wall = time.time()  # for adapter-log time-window filtering
    _LOGGER.info(
        "[r=%d/%d] sandbox.start  container=%s  chain=%s%s",
        t, rounds, sandbox_container, list(s_prime_box.chain),
        "  (chain rewritten)" if forced else "",
    )
    hp, _tc, host_chain_dir = _seams.sandbox_run(
        s_prime_box, list(s_prime_box.chain),
        target_model, sandbox_container, sandbox_run_fn,
    )
    sb_dt = time.monotonic() - sb_t0
    _io.snapshot_chain(paths.chains_dir, t, s_prime, host_chain_dir)

    bypassed = _bypass_from_findings(audit_findings) if not gated else False
    # Wrap hp with metadata so the cross-skill judge can read
    # ``distinct_skills_exec`` (slotted SensorReading has no metadata
    # field; see ``red_team.oracle.sensor_metadata``).  Other judges
    # ignore the wrapper since attribute access falls through.
    from red_team.oracle.sensor_metadata import with_skill_metadata
    reading = with_skill_metadata(hp, chain=list(s_prime.chain or []))
    try:
        lethal_flag = bool(oracle_fn(s_prime, reading, s_prime.o))
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("oracle raised %s; treating as non-lethal", exc)
        lethal_flag = False
    _LOGGER.info(
        "[r=%d/%d] sandbox.done   tool_calls=%d  cmds=%d  files_mod=%d  "
        "files_del=%d  lethal=%s  elapsed=%.1fs",
        t, rounds,
        len(getattr(hp, "tool_calls", []) or []),
        len(getattr(hp, "commands_executed", []) or []),
        len(getattr(hp, "files_modified", []) or []),
        len(getattr(hp, "files_deleted", []) or []),
        lethal_flag, sb_dt,
    )

    if gated and lethal_flag:
        audit_t0 = time.monotonic()
        _LOGGER.info(
            "[r=%d/%d] audit.start  blue=%s  chain=%s  (gated: lethal)",
            t, rounds, blue_team, list(s_prime.chain or []),
        )
        audit_findings = _seams.call_blue_team(
            s_prime, repo, blue_team, blue_team_fn,
            sandbox_container=sandbox_container,
        )
        audit_dt = time.monotonic() - audit_t0
        bypassed = _bypass_from_findings(audit_findings)
        state.last_audit_findings = list(audit_findings)
        _LOGGER.info(
            "[r=%d/%d] audit.done   verdict=%s  findings=%d  elapsed=%.1fs",
            t, rounds,
            "APPROVE" if bypassed else "REJECT",
            len(audit_findings or []), audit_dt,
        )

    target_reply = extract_target_reply(
        sandbox_container,
        chain_fingerprint=list(s_prime.chain or []),
        since_ts=sb_t0_wall,
    )
    from red_team.external_baselines.trace import VictimTrace
    state.last_victim_trace = VictimTrace.from_sensor_reading(
        hp,
        final_output=target_reply,
        tool_calls=tuple(
            str(tc.get("tool", tc)) for tc in (_tc or [])[:50]
        ),
    )
    state.last_runtime_findings = synthesize_runtime_feedback(
        hp, lethal_flag=lethal_flag, round_index=t,
        target_reply=target_reply,
    )

    if bypassed:
        state.total_bypass += 1
        if state.first_bypass_round is None:
            state.first_bypass_round = t
    if lethal_flag:
        state.total_lethal += 1

    _io.write_log_row(paths.log, _io.build_log_row(
        round_index=t, variant=s_prime, findings=audit_findings, hp=hp,
        bypassed=bypassed, lethal_flag=lethal_flag, backend=backend_tag,
        stop_reason=str(getattr(ar, "stop_reason", "") or ""),
        rationale=str(getattr(ar, "rationale", "") or ""),
        tool_chain=list(getattr(ar, "tool_chain", []) or []),
        cost_usd=float(getattr(ar, "cost_usd", 0.0) or 0.0),
        is_expansion=False, phase=search_space,
        forced_topology=forced, chain_original=chain_orig, chain_used=chain_used,
    ))
    _io.append_attack_log(
        paths.attack_log, round_index=t, seed=seed, variant=s_prime,
        bypassed=bypassed, lethal_flag=lethal_flag,
        rationale=str(getattr(ar, "rationale", "") or ""),
        tool_chain=list(getattr(ar, "tool_chain", []) or []),
        is_expansion=False,
    )

    try:
        harvest_row = build_main_round_harvest_row(
            run_id=run_id, seed_id=seed_id,
            round_index=t, stage=search_space,
            parent_variant=state.current_seed,
            s_prime=s_prime,
            parent_bypass=bool(bypassed),
            parent_findings=audit_findings,
            lethal=bool(lethal_flag),
            target_model=target_model,
            tool_chain=list(getattr(ar, "tool_chain", []) or []),
            sandbox_files_modified=(
                list(getattr(hp, "files_modified", []) or [])
                + list(getattr(hp, "files_deleted", []) or [])
            ),
            sandbox_commands_executed=list(
                getattr(hp, "commands_executed", []) or []
            ),
        )
        append_harvest_row(paths.harvest, harvest_row)
        if memory_runtime.enabled:
            memory_runtime.on_harvest(harvest_row)
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("harvest write (main round) failed: %s", exc)

    star = bool(bypassed and lethal_flag)
    _LOGGER.info(
        "[r=%d/%d] outcome  bypass=%s  lethal=%s  %s",
        t, rounds, bypassed, lethal_flag,
        "* HIT" if star else "* no",
    )

    # Surface this round's outcome to the next round's prompt — the
    # chain we audited (parent), the mutation we produced, and the
    # sandbox result.  ``audit_findings`` here is the verdict on the
    # parent into THIS round (slightly off-by-one, see the prompt
    # template note); the summary's strategy hint helps the LLM not
    # over-engineer a working mutation.
    sandbox_evidence = list(getattr(hp, "evidence", []) or [])
    state.last_round_summary = build_last_round_summary(
        round_index=t,
        parent_chain=list((seed.chain if t == 1 else state.current_seed.chain) or []),
        audit_findings=(
            [] if run_config.feedback_mode == "scalar" else list(audit_findings)
        ),
        bypassed=None if gated and not lethal_flag else bypassed,
        s_prime=s_prime,
        lethal_flag=lethal_flag,
        lethal_evidence=sandbox_evidence,
    )

    # Always advance current_seed to the freshly produced winner so
    # phase-2 (or _dump_phase1_winner) reads the canonical winner from
    # state, regardless of whether ★ fired.
    state.current_seed = s_prime
    if star:
        state.star_round = t
        return True
    return False


def run_phase1(
    *,
    paths: _RunPaths,
    state: _RunState,
    run_config: RunConfig,
    search_space: str,
    rounds: int,
    blue_team: str,
    blue_team_fn: Callable[..., list[PaperFinding]] | None,
    sandbox_container: str,
    target_model: str,
    mutator_model: str,
    agent_run_fn: Callable[..., Any],
    oracle_fn: Callable[..., bool],
    sandbox_run_fn: Callable[..., Any] | None,
    seed: SkillVariant,
    repo: RepositorySlice,
    run_id: str,
    memory_runtime: MemoryRuntime | None = None,
) -> Path:
    """Run phase-1 main loop for up to ``rounds`` rounds.

    Returns the run directory.  Writes the terminal ``summary.json``
    using the values accumulated on ``state``; the runner reads that
    summary unchanged when phase=="main_only" and may augment it (or
    leave it) when phase=="both".
    """
    seed_id = str(seed.metadata.get("seed_id") or seed.content_hash())

    if memory_runtime is None:
        memory_runtime = create_memory_runtime(
            memory_db=paths.memory_db, run_config=run_config
        )

    for t in range(1, rounds + 1):
        state.rounds_ran = t
        star = _run_one_round(
            t,
            state=state, paths=paths, run_config=run_config,
            search_space=search_space, rounds=rounds,
            blue_team=blue_team, blue_team_fn=blue_team_fn,
            sandbox_container=sandbox_container,
            target_model=target_model,
            agent_run_fn=agent_run_fn, oracle_fn=oracle_fn,
            sandbox_run_fn=sandbox_run_fn,
            seed=seed, repo=repo, run_id=run_id, seed_id=seed_id,
            memory_runtime=memory_runtime,
        )
        if star or state.strategy_early_stop:
            break

    _io.write_summary(paths.summary, _io.build_summary(
        run_id=run_id,
        seed_id=seed_id,
        repo_id=repo.repo_id, target_model=target_model,
        mutator_model=mutator_model, rounds_ran=state.rounds_ran,
        first_bypass_round=state.first_bypass_round,
        total_bypass=state.total_bypass, total_lethal=state.total_lethal,
        total_cost=state.total_cost, run_config=run_config,
        expansion_audits=state.expansion_audits,
        expansion_audit_cache_hits=state.expansion_audit_cache_hits,
    ))
    emit_run_complete(
        _LOGGER,
        run_id=run_id,
        rounds_ran=state.rounds_ran,
        first_bypass_round=state.first_bypass_round,
        total_bypass=state.total_bypass,
        total_lethal=state.total_lethal,
        total_cost_usd=state.total_cost,
        star_round=state.star_round,
    )
    return paths.run_dir
