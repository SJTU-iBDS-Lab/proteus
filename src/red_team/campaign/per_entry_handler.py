"""Per-entry checkpoint handler for S2/S3 expansion rows.

Single-instance state machine used by phase-2 expansion at every entry
point that produces a candidate variant:

- audits the mutation via the blue team (with content-hash dedup);
- flushes one log.jsonl row per entry (no batch buffering — disk
  durability matters for crash recovery);
- appends a HarvestRow for the offline extractor when ``harvest_path``
  is configured;
- triggers the memory runtime hook so phase-2 evolving-memory mode
  picks up new units immediately.

Instantiate **once per expansion call site** (per ★ winner / per
expansion-only winner) and pass ``on_path_result`` /
``on_surface_executed`` as callbacks to the expansion functions.

Returned ``finalize()`` yields ``(audits_issued, cache_hits)`` so the
caller can record per-seed telemetry on the shared :class:`_RunState`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from core.models import RepositorySlice
from core.schemas import PaperFinding, SkillVariant
from red_team.campaign import io as _io
from red_team.campaign import seams as _seams
from red_team.campaign.state import _bypass_from_findings
from red_team.log.harvest_writer import (
    append_harvest_row,
    build_expansion_harvest_row,
)
from red_team.log.memory.runtime_hook import MemoryRuntime

__all__ = ["_PerEntryHandler"]

_LOGGER = logging.getLogger(__name__)


class _PerEntryHandler:
    """Audit + log + harvest + memory hook for one expansion call site."""

    def __init__(
        self,
        *,
        repo: RepositorySlice,
        blue_team: str,
        blue_team_fn: Callable[..., list[PaperFinding]] | None,
        sandbox_container: str,
        log_path: Path,
        harvest_path: Path | None,
        backend_tag: str,
        run_id: str,
        seed_id: str,
        parent_variant: SkillVariant,
        target_model: str,
        memory_runtime: MemoryRuntime | None,
    ) -> None:
        self._repo = repo
        self._blue_team = blue_team
        self._blue_team_fn = blue_team_fn
        self._sandbox_container = sandbox_container
        self._log_path = log_path
        self._harvest_path = harvest_path
        self._backend_tag = backend_tag
        self._run_id = run_id
        self._seed_id = seed_id
        self._parent_variant = parent_variant
        self._target_model = target_model
        self._memory_runtime = memory_runtime
        self._audit_cache: dict[str, list[PaperFinding]] = {}
        self._audits = 0
        self._hits = 0

    def _verdict(self, mutation: SkillVariant) -> bool:
        cache_key = mutation.content_hash()
        cached = self._audit_cache.get(cache_key)
        if cached is not None:
            self._hits += 1
            findings = cached
        else:
            findings = _seams.call_blue_team(
                mutation, self._repo, self._blue_team, self._blue_team_fn,
                sandbox_container=self._sandbox_container,
            )
            self._audit_cache[cache_key] = list(findings)
            self._audits += 1
        return _bypass_from_findings(findings)

    def _flush_one(self, entry: Any, stage: str) -> None:
        try:
            entry.bypassed = self._verdict(entry.mutation)
        except Exception as exc:
            _LOGGER.warning("per-entry audit (%s) failed: %s", stage, exc)
            entry.bypassed = None
        try:
            _io.write_log_row(self._log_path, _io.build_log_row(
                round_index=None, variant=entry.mutation, findings=[],
                hp=entry.hp, bypassed=entry.bypassed,
                lethal_flag=bool(entry.lethal),
                backend=self._backend_tag, stop_reason="",
                rationale=str(getattr(entry, "rationale", "") or ""),
                tool_chain=list(getattr(entry, "tool_chain", []) or []),
                cost_usd=0.0, is_expansion=True, phase=stage,
            ))
        except Exception as exc:
            _LOGGER.warning("per-entry log row (%s) failed: %s", stage, exc)
        if self._harvest_path is None:
            return
        try:
            row = build_expansion_harvest_row(
                run_id=self._run_id, seed_id=self._seed_id,
                parent_variant=self._parent_variant, entry=entry,
                stage=stage, target_model=self._target_model,
            )
            append_harvest_row(self._harvest_path, row)
            if self._memory_runtime is not None and self._memory_runtime.enabled:
                self._memory_runtime.on_harvest(row)
        except Exception as exc:
            _LOGGER.warning("per-entry harvest (%s) failed: %s", stage, exc)

    def on_path_result(self, entry: Any) -> None:
        self._flush_one(entry, "path_expansion")

    def on_surface_executed(self, entry: Any) -> None:
        self._flush_one(entry, "surface_expansion")

    def finalize(self) -> tuple[int, int]:
        """Return ``(audits_issued, cache_hits)`` accumulated across all calls."""
        return self._audits, self._hits
