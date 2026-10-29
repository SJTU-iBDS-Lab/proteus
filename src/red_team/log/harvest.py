"""``HarvestRow`` — the extractor's input contract.

One row per audited variant (post per-entry audit so ``bypass`` is
always concrete). The runner appends rows to
``outputs/evolution/<run_id>/harvest.jsonl``; the offline
:mod:`red_team.log.memory.extractor` consumes them.

Plan §3.4 / §3.5 is the source of truth for fields and ordering.
The hard invariant — ``bypass`` is never ``None`` — is enforced in
:meth:`__post_init__` so any caller that bypasses the per-entry
audit step raises immediately rather than poisoning the extraction
batch downstream.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


_VALID_STAGES: frozenset[str] = frozenset(
    {"evasion", "path_expansion", "surface_expansion"}
)


@dataclass(frozen=True, slots=True)
class HarvestRow:
    """One audited-variant row appended to ``harvest.jsonl``.

    The DTO is deliberately self-contained: extractor reads a single
    line and has every field it needs (variant snapshot, audit
    findings, sandbox trace, provenance) without touching live
    objects or joining other JSONL streams.
    """

    # —— Identity ——
    run_id: str
    seed_id: str
    round: int
    stage: Literal["evasion", "path_expansion", "surface_expansion"]
    timestamp: str

    # —— SkillVariant snapshot ——
    variant: dict[str, Any]
    variant_hash: str
    parent_variant_hash: str | None

    # —— Audit (post per-entry audit; bypass never None) ——
    bypass: bool
    audit_findings: tuple[dict[str, Any], ...]
    audit_source: str

    # —— Sandbox ——
    lethal: bool
    target_model: str
    tool_chain: tuple[dict[str, Any], ...]
    sandbox_trace_summary: str
    sandbox_files_modified: tuple[str, ...]
    sandbox_commands_executed: tuple[str, ...]

    # —— Provenance ——
    family_candidate: dict[str, Any] | None
    s3_hint: dict[str, Any] | None
    refused: bool

    def __post_init__(self) -> None:
        # Phase 1/2 contract: the writer must run after the per-entry
        # audit hook so bypass is concrete. NULL/None here breaks
        # downstream extraction silently — fail loud at construction
        # instead.
        if not isinstance(self.bypass, bool):
            raise ValueError(
                "HarvestRow.bypass must be a concrete bool — "
                "the writer must run after the per-entry audit hook"
            )
        if self.stage not in _VALID_STAGES:
            raise ValueError(
                f"HarvestRow.stage={self.stage!r} not in "
                f"{sorted(_VALID_STAGES)}"
            )
        if not self.run_id or not self.seed_id or not self.variant_hash:
            raise ValueError(
                "HarvestRow.run_id / seed_id / variant_hash must all be "
                "non-empty"
            )

    # ------------------------------------------------------------------ #
    # Serialisation                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary (tuples → lists)."""
        return asdict(self)

    def to_jsonl_line(self) -> str:
        """One compact JSON line ready for ``harvest.jsonl``."""
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HarvestRow":
        return cls(
            run_id=d["run_id"],
            seed_id=d["seed_id"],
            round=int(d["round"]),
            stage=d["stage"],
            timestamp=d["timestamp"],
            variant=dict(d.get("variant", {})),
            variant_hash=d["variant_hash"],
            parent_variant_hash=d.get("parent_variant_hash"),
            bypass=bool(d["bypass"]),
            audit_findings=tuple(d.get("audit_findings", ())),
            audit_source=d["audit_source"],
            lethal=bool(d["lethal"]),
            target_model=d["target_model"],
            tool_chain=tuple(d.get("tool_chain", ())),
            sandbox_trace_summary=d.get("sandbox_trace_summary", ""),
            sandbox_files_modified=tuple(d.get("sandbox_files_modified", ())),
            sandbox_commands_executed=tuple(d.get("sandbox_commands_executed", ())),
            family_candidate=d.get("family_candidate"),
            s3_hint=d.get("s3_hint"),
            refused=bool(d.get("refused", False)),
        )

    @classmethod
    def from_jsonl_line(cls, line: str) -> "HarvestRow":
        return cls.from_dict(json.loads(line))


__all__ = ["HarvestRow"]
