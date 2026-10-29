"""Append-only writer for ``harvest.jsonl`` (the extractor's input).

Two construction helpers cover the two writer call-sites:

* :func:`build_main_round_harvest_row` — used by the runner inside the
  main loop right after the per-round log row.  The audit verdict
  carried by this row is **the parent's** audit (off-by-one bug
  documented in ``docs/known_bugs.md §1``).  We tag ``audit_source``
  with ``"skillvetter:parent_audit"`` so downstream extractors can
  reason about its provenance.
* :func:`build_expansion_harvest_row` — used by the runner's per-entry
  handler (``_PerEntryHandler``) for path / surface entries.
  Their ``bypassed`` value is the result of the per-entry re-audit, so
  ``audit_source`` is ``"skillvetter:expansion_reaudit"``.

The actual append call (:func:`append_harvest_row`) re-asserts the
``HarvestRow.bypass is not None`` invariant before flushing — failing
loud rather than letting a silent ``None`` poison the offline
extraction batch.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, cast

from core.schemas import PaperFinding, SkillVariant
from red_team.log.harvest import HarvestRow

__all__ = [
    "append_harvest_row",
    "build_main_round_harvest_row",
    "build_expansion_harvest_row",
    "iso_now",
]


def iso_now() -> str:
    """ISO-8601 UTC timestamp with second resolution (matches existing logs)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _findings_to_tuple(findings: Iterable[Any] | None) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for f in findings or ():
        if isinstance(f, PaperFinding):
            out.append(f.to_dict())
        elif isinstance(f, dict):
            out.append(dict(f))
        else:
            # Best-effort attribute scrape for foreign finding shapes.
            out.append(
                {
                    "finding_id": getattr(f, "finding_id", ""),
                    "layer": getattr(f, "layer", ""),
                    "reason_code": getattr(f, "reason_code", ""),
                    "severity": getattr(f, "severity", ""),
                    "explanation": getattr(f, "explanation", ""),
                }
            )
    return tuple(out)


def _tool_chain_to_tuple(tool_chain: Iterable[Any] | None) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for tc in tool_chain or ():
        if isinstance(tc, dict):
            out.append(dict(tc))
        elif isinstance(tc, str):
            out.append({"name": tc})
        else:
            out.append({"name": str(getattr(tc, "name", tc))})
    return tuple(out)


def _safe_str_tuple(items: Iterable[Any] | None) -> tuple[str, ...]:
    return tuple(str(x) for x in (items or ()))


def append_harvest_row(path: Path | str, row: HarvestRow) -> None:
    """Append a single :class:`HarvestRow` as one JSONL line.

    ``HarvestRow.__post_init__`` already rejects ``bypass=None``; we
    re-check defensively here so a rebuild that bypasses the dataclass
    constructor still cannot persist a malformed row.
    """
    if row.bypass is None:  # type: ignore[unreachable]
        raise ValueError(
            "harvest row bypass must be a concrete bool — "
            "writer must run after the per-entry audit hook"
        )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(row.to_jsonl_line())
        fh.write("\n")


_LEGACY_STAGE_ALIASES = {
    # The runner's default ``search_space`` is ``free_mutation``; the
    # canonical stage vocabulary in HarvestRow / CampaignLogRow only
    # admits the three semantic names.  Map legacy → canonical at
    # write-site so HarvestRow.__post_init__ stays strict.
    "free_mutation": "evasion",
    "S1": "evasion",
    "S2": "path_expansion",
    "S3": "surface_expansion",
}


_StageT = Literal["evasion", "path_expansion", "surface_expansion"]


def _normalize_stage(stage: str) -> _StageT:
    return cast(_StageT, _LEGACY_STAGE_ALIASES.get(stage, stage))


def build_main_round_harvest_row(
    *,
    run_id: str,
    seed_id: str,
    round_index: int,
    stage: str,
    parent_variant: SkillVariant,
    s_prime: SkillVariant,
    parent_bypass: bool,
    parent_findings: Iterable[Any] | None,
    lethal: bool,
    target_model: str,
    tool_chain: Iterable[Any] | None,
    sandbox_files_modified: Iterable[Any] | None,
    sandbox_commands_executed: Iterable[Any] | None,
    sandbox_trace_summary: str = "",
    refused: bool = False,
    timestamp: str | None = None,
    audit_source: str = "skillvetter:parent_audit",
) -> HarvestRow:
    """Build a HarvestRow for a main-round (free_mutation / evasion) event.

    The ``parent_bypass`` / ``parent_findings`` arguments encode the
    off-by-one-buggy semantics of ``log.jsonl`` that we have to live
    with for now; the ``audit_source`` tag makes the provenance
    explicit so a future re-audit pass can replace these with the true
    s_prime verdict without changing the file format.
    """
    return HarvestRow(
        run_id=run_id,
        seed_id=seed_id,
        round=int(round_index),
        stage=_normalize_stage(stage),
        timestamp=timestamp or iso_now(),
        variant=s_prime.to_dict(),
        variant_hash=s_prime.content_hash(),
        parent_variant_hash=parent_variant.content_hash(),
        bypass=bool(parent_bypass),
        audit_findings=_findings_to_tuple(parent_findings),
        audit_source=audit_source,
        lethal=bool(lethal),
        target_model=target_model,
        tool_chain=_tool_chain_to_tuple(tool_chain),
        sandbox_trace_summary=sandbox_trace_summary,
        sandbox_files_modified=_safe_str_tuple(sandbox_files_modified),
        sandbox_commands_executed=_safe_str_tuple(sandbox_commands_executed),
        family_candidate=None,
        s3_hint=None,
        refused=bool(refused),
    )


def build_expansion_harvest_row(
    *,
    run_id: str,
    seed_id: str,
    parent_variant: SkillVariant,
    entry: Any,
    stage: str,
    target_model: str,
    timestamp: str | None = None,
    audit_source: str = "skillvetter:expansion_reaudit",
) -> HarvestRow:
    """Build a HarvestRow from a path/surface expansion entry.

    Each entry exposes ``mutation`` (SkillVariant), ``bypassed`` (bool —
    written back by the runner's per-entry audit hook), ``lethal``,
    ``hp`` (sandbox result, may be None), plus optional ``rationale`` /
    ``tool_chain`` / ``family_candidate`` / ``s3_hint`` attributes.
    """
    mutation = entry.mutation
    bypassed = entry.bypassed
    if bypassed is None:
        raise ValueError(
            f"expansion entry for variant={mutation.content_hash()} has "
            "bypassed=None — per-entry audit hook must run before harvest write"
        )
    hp = getattr(entry, "hp", None)
    files_modified = getattr(hp, "files_modified", []) if hp is not None else []
    files_deleted = getattr(hp, "files_deleted", []) if hp is not None else []
    commands = getattr(hp, "commands_executed", []) if hp is not None else []
    sandbox_files_modified = list(files_modified) + list(files_deleted)
    summary = (
        f"hp.files_modified={len(files_modified)} "
        f"hp.files_deleted={len(files_deleted)} "
        f"hp.commands_executed={len(commands)}"
        if hp is not None
        else "no_sandbox_result"
    )
    return HarvestRow(
        run_id=run_id,
        seed_id=seed_id,
        round=int(getattr(entry, "round", 0) or 0),
        stage=_normalize_stage(stage),
        timestamp=timestamp or iso_now(),
        variant=mutation.to_dict(),
        variant_hash=mutation.content_hash(),
        parent_variant_hash=parent_variant.content_hash(),
        bypass=bool(bypassed),
        audit_findings=_findings_to_tuple(getattr(entry, "audit_findings", None)),
        audit_source=audit_source,
        lethal=bool(getattr(entry, "lethal", False)),
        target_model=target_model,
        tool_chain=_tool_chain_to_tuple(getattr(entry, "tool_chain", None)),
        sandbox_trace_summary=summary,
        sandbox_files_modified=_safe_str_tuple(sandbox_files_modified),
        sandbox_commands_executed=_safe_str_tuple(commands),
        family_candidate=getattr(entry, "family_candidate", None),
        s3_hint=getattr(entry, "s3_hint", None),
        refused=False,
    )
