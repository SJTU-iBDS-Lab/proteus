"""Strict DTOs for the red-team log streams.

Three immutable dataclasses describe the three JSONL streams the red team
writes under ``outputs/evolution/<run_id>/``:

* :class:`CampaignLogRow`  — ``log.jsonl``     (single-round events)
* :class:`AttackLogRow`    — ``attack_log.jsonl`` (cross-round memory,
  indexed by BM25 for retrieval)
* :class:`SummaryRow`      — ``summary.json``  (terminal rollup)

Stage naming
------------
``CampaignLogRow.stage`` must be one of the three *semantic* names
``evasion`` / ``path_expansion`` / ``surface_expansion``. The legacy
positional identifiers (``S1`` / ``S2`` / ``S3``) are rejected at
construction time to prevent the abbreviation from leaking into new
code paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CampaignLogRow:
    """A single row appended to ``outputs/evolution/<run_id>/log.jsonl``.

    This is what the evolution driver used to write inline — one JSON line
    per round event (either a per-round attack event or an expansion
    event that sits between rounds).

    ``stage`` must use the semantic vocabulary; the legacy ``S1``/``S2``/
    ``S3`` abbreviations are rejected.
    """

    run_id: str
    round: int | None                   # None for expansion rows.
    stage: str                          # Semantic: evasion / path_expansion / surface_expansion.
    is_expansion: bool
    timestamp: str                      # ISO-8601.
    seed_id: str
    repo_id: str
    target_model: str
    bypass: bool
    lethal: bool
    stop_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    _VALID_STAGES = frozenset({"evasion", "path_expansion", "surface_expansion"})

    def __post_init__(self) -> None:
        if self.stage not in self._VALID_STAGES:
            raise ValueError(
                f"invalid stage {self.stage!r}; must be one of "
                f"{sorted(self._VALID_STAGES)}"
            )


@dataclass(frozen=True, slots=True)
class SummaryRow:
    """Terminal summary written to ``outputs/evolution/<run_id>/summary.json``."""

    run_id: str
    rounds_executed: int
    first_bypass_round: int | None
    first_lethal_round: int | None
    total_cost_usd: float
    stop_reason: str
    terminated_at: str                  # ISO-8601.


@dataclass(frozen=True, slots=True)
class AttackLogRow:
    """A single row of ``outputs/evolution/<run_id>/attack_log.jsonl``.

    Cross-round memory used for BM25 retrieval across rounds.
    ``findings_summary`` is the pre-assembled text blob (reason +
    category + reason_code concatenation) that feeds the BM25 index.
    """

    run_id: str
    round: int
    seed_id: str
    repo_id: str
    variant_hash: str
    findings_summary: str
    tool_chain: tuple[str, ...]
    bypass: bool
    lethal: bool
    timestamp: str


__all__ = ["CampaignLogRow", "SummaryRow", "AttackLogRow"]
