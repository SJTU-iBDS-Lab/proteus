"""Log row / summary assembly + JSONL writers for the campaign runner.

This module owns the dict-schema used by ``log.jsonl`` and the terminal
``summary.json``.  The runner hands it primitive values (variant,
findings, honeypot result, ...) and receives a fully-baked ``dict`` that
is byte-equal to what the pre-split driver used to emit.

Two invariants:

* Field names, ordering within a row, and ``None`` vs missing semantics
  are pinned by the unit tests in ``tests/unit/red_team/campaign/test_io.py``
  and the smoke suite under ``tests/unit/evolution/test_driver_smoke.py``.
* All persistence goes through :mod:`red_team.log.campaign_log` /
  :mod:`red_team.log.summary` so the on-disk bytes remain canonical.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.models import RepositorySlice
from core.schemas import PaperFinding, SkillVariant
from red_team.campaign.run_config import RunConfig
from red_team.log import campaign_log as _campaign_log
from red_team.log import summary as _summary_mod
from red_team.log.attack_log import append as _attack_log_append
from sandbox.honeypot import HoneypotResult

_LOGGER = logging.getLogger(__name__)


@dataclass
class _RunPaths:
    """Resolved output paths for a single run."""
    run_dir: Path
    log: Path
    attack_log: Path
    summary: Path
    mutations_dir: Path
    chains_dir: Path
    harvest: Path
    memory_db: Path


def _default_run_id(seed: SkillVariant, repo: RepositorySlice, target_model: str) -> str:
    seed_id = str(seed.metadata.get("seed_id") or seed.content_hash())
    repo_id = repo.repo_id or "repo"
    model_id = (target_model or "model").replace("/", "_")
    return f"{seed_id}_{repo_id}_{model_id}_{secrets.token_hex(4)}"


def _resolve_paths(output_root: Path | str, run_id: str) -> _RunPaths:
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    mutations = run_dir / "mutations"
    mutations.mkdir(parents=True, exist_ok=True)
    chains = run_dir / "chains"
    chains.mkdir(parents=True, exist_ok=True)
    return _RunPaths(
        run_dir=run_dir,
        log=run_dir / "log.jsonl",
        attack_log=run_dir / "attack_log.jsonl",
        summary=run_dir / "summary.json",
        mutations_dir=mutations,
        chains_dir=chains,
        harvest=run_dir / "harvest.jsonl",
        memory_db=run_dir / "memory.sqlite",
    )


def _dump_phase1_winner(
    paths: _RunPaths,
    winner: SkillVariant,
    seed_id: str,
    target_model: str,
) -> Path:
    """Save the phase-1 winner SkillVariant for a phase-2 follow-up.

    Phase-2 ``expansion_only`` mode reads this file to know which
    variant to expand from.  ``winner`` is ``state.current_seed`` at
    end of the main loop -- either the ★ s_prime (when the loop
    broke on first ★) or the last s_prime if no ★ ever fired.

    Returns the dumped winner path so the caller can pass it to a
    phase-2 barrier coordinator.
    """
    payload = {
        "seed_id": seed_id,
        "target_model": target_model,
        "winner": winner.to_dict(),
        "winner_hash": winner.content_hash(),
    }
    path = paths.run_dir / "phase1_winner.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def _now_iso() -> str:
    """UTC ISO-8601 timestamp used across log rows."""
    return _campaign_log.now_iso()


def _coerce_tool_chain(tool_chain: Any) -> list[Any]:
    """Normalize a tool_chain payload to a plain list for JSON logging."""
    if tool_chain is None:
        return []
    if isinstance(tool_chain, list):
        return list(tool_chain)
    return [tool_chain]


def build_log_row(
    *,
    round_index: int | None,
    variant: SkillVariant | None,
    findings: list[PaperFinding],
    hp: HoneypotResult | None,
    bypassed: bool | None,
    lethal_flag: bool,
    backend: str,
    stop_reason: str,
    rationale: str,
    tool_chain: list[Any],
    cost_usd: float,
    is_expansion: bool,
    phase: str,
    forced_topology: bool = False,
    chain_original: list[str] | None = None,
    chain_used: list[str] | None = None,
) -> dict[str, Any]:
    """Construct a single ``log.jsonl`` row."""
    row: dict[str, Any] = {
        "round": round_index,
        "variant_hash": variant.content_hash() if variant is not None else None,
        "o": variant.o.value if variant is not None else None,
        "tau": variant.tau.value if variant is not None else None,
        "chain": list(variant.chain) if variant is not None else None,
        # ``bypassed`` is tri-state for expansion rows: True/False once
        # the per-entry audit hook fires, None when no hook was wired
        # (test-only path; production runner always provides one).
        # Main-round rows always pass a real bool, so the column is
        # boolean-typed in practice — None only appears for unaudited
        # expansion rows.
        "bypassed": bypassed if bypassed is None else bool(bypassed),
        "lethal": bool(lethal_flag),
        "backend": backend,
        "stop_reason": stop_reason,
        "rationale": rationale or "",
        "tool_chain": _coerce_tool_chain(tool_chain),
        "findings_count": len(findings or []),
        "findings": [
            {
                "finding_id": getattr(f, "finding_id", ""),
                "layer": getattr(f, "layer", ""),
                "reason_code": getattr(f, "reason_code", ""),
                "severity": getattr(f, "severity", ""),
                "category": getattr(f, "category", ""),
                "explanation": (getattr(f, "explanation", "") or "")[:400],
                "origin_skill": (getattr(f, "metadata", {}) or {}).get("origin_skill"),
                "red_flags": (getattr(f, "metadata", {}) or {}).get("red_flags"),
                "risk_level": (getattr(f, "metadata", {}) or {}).get("risk_level"),
            }
            for f in (findings or [])
        ],
        "cost_usd": float(cost_usd or 0.0),
        "is_expansion": bool(is_expansion),
        "phase": phase,
        "forced_topology": bool(forced_topology),
        "chain_original": list(chain_original) if chain_original is not None else None,
        "chain_used": list(chain_used) if chain_used is not None else None,
        "hp": hp.to_dict() if hp is not None else None,
        "timestamp": _now_iso(),
    }
    return row


def write_log_row(log_path: Path, row: dict[str, Any]) -> None:
    """Append a row to ``log.jsonl`` via :mod:`red_team.log.campaign_log`."""
    _campaign_log.write_row(log_path, row)


def append_attack_log(
    attack_log_path: Path,
    *,
    round_index: int | None,
    seed: SkillVariant,
    variant: SkillVariant | None,
    bypassed: bool,
    lethal_flag: bool,
    rationale: str,
    tool_chain: list[Any],
    is_expansion: bool,
) -> None:
    """Append a row to the attack log (used by BM25 search)."""
    seed_id = str(seed.metadata.get("seed_id") or seed.content_hash())
    entry: dict[str, Any] = {
        "round": round_index if round_index is not None else -1,
        "seed_id": seed_id,
        "variant_hash": variant.content_hash() if variant is not None else None,
        "o": variant.o.value if variant is not None else None,
        "tau": variant.tau.value if variant is not None else None,
        "bypassed": bool(bypassed),
        "lethal": bool(lethal_flag),
        "reason": rationale or "",
        "reason_codes_evaded": [],
        "tool_chain_summary": _coerce_tool_chain(tool_chain),
        "is_expansion": bool(is_expansion),
        "timestamp": _now_iso(),
    }
    _attack_log_append(attack_log_path, entry)


def build_summary(
    *,
    run_id: str,
    seed_id: str,
    repo_id: str | None,
    target_model: str,
    mutator_model: str,
    rounds_ran: int,
    first_bypass_round: int | None,
    total_bypass: int,
    total_lethal: int,
    total_cost: float,
    run_config: RunConfig,
    expansion_audits: int = 0,
    expansion_audit_cache_hits: int = 0,
) -> dict[str, Any]:
    """Build the terminal ``summary.json`` payload."""
    return {
        "run_id": run_id,
        "seed_id": seed_id,
        "repo_id": repo_id,
        "target_model": target_model,
        "mutator_model": mutator_model,
        "rounds_ran": rounds_ran,
        "first_bypass_round": first_bypass_round,
        "total_bypass": total_bypass,
        "total_lethal": total_lethal,
        "total_cost_usd": round(total_cost, 6),
        "ablation": dataclasses.asdict(run_config),
        "expansion_audits": int(expansion_audits),
        "expansion_audit_cache_hits": int(expansion_audit_cache_hits),
    }


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    """Persist ``summary.json`` via :mod:`red_team.log.summary`."""
    _summary_mod.write_summary(summary_path, summary)


def persist_mutation_md(
    mutations_dir: Path,
    round_index: int,
    s_prime: SkillVariant,
    skill_root: Path,
) -> None:
    """Write the full SKILL.md the sandbox will see, before the sandbox call.

    Assembled from the benign body of the sink skill plus the mutator's
    payload segment.  Written BEFORE the sandbox call so the artifact
    survives a sandbox crash.
    """
    chain_list = list(s_prime.chain)
    pidx = s_prime.poison_index
    if pidx < 0:
        pidx += len(chain_list)
    pidx = max(0, min(len(chain_list) - 1, pidx))
    benign_body = ""
    if chain_list:
        orig_md = skill_root / chain_list[pidx] / "SKILL.md"
        if orig_md.is_file():
            benign_body = orig_md.read_text(encoding="utf-8", errors="replace")
    separator = (
        ""
        if benign_body.endswith("\n\n")
        else ("\n\n" if benign_body.endswith("\n") else "\n\n")
    )
    full = benign_body + separator + (s_prime.d or "").lstrip("\n")
    (mutations_dir / f"R{round_index:02d}_{s_prime.content_hash()}.md").write_text(
        full, encoding="utf-8"
    )


def snapshot_chain(
    chains_dir: Path,
    round_index: int,
    s_prime: SkillVariant,
    host_chain_dir: Path | None,
) -> None:
    """Copy the host poisoned-chain tree into ``run_dir/chains/``."""
    if host_chain_dir is None or not host_chain_dir.is_dir():
        return
    import shutil as _shutil

    snap = chains_dir / f"R{round_index:02d}_{s_prime.content_hash()}"
    if snap.exists():
        _shutil.rmtree(snap)
    try:
        _shutil.copytree(host_chain_dir, snap)
    except OSError as exc:  # pragma: no cover - defensive
        _LOGGER.warning("chain snapshot failed for R%d: %s", round_index, exc)
