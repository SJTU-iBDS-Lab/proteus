"""Runner-side glue for the M2 (``evolving``) memory mode.

Called at three places in :mod:`red_team.campaign.runner` — after each
``HarvestRow`` write (main / path_expansion / surface_expansion).  In
M0 / M1 mode the hook is a no-op; in M2 it:

1. Extracts a :class:`MemoryUnit` from the harvest row (3 sub-prompts).
2. Upserts the new unit into the run's per-run ``memory.sqlite``.
3. Calls :func:`red_team.log.memory.evolver.evolve` to attempt
   higher-order family abstraction across same-(seed, target)
   neighbours.

All LLM calls are fail-safe (timeout / parse / API error logged but
never crash the run; the run's harvest.jsonl remains the
authoritative ground truth).

The hook owns its own LLM client + embedder + Store handle, lazily
initialised on first call and reused for the run's lifetime.  This
mirrors the runner's existing pattern of lazy ``_seams`` resolution
and keeps the runner.py diff minimal.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from red_team.campaign.run_config import RunConfig
from red_team.log.harvest import HarvestRow
from red_team.log.memory.evolver import EvolveConfig, evolve
from red_team.log.memory.extractor import (
    DEFAULT_EXTRACTOR_VERSION,
    ExtractionError,
    extract_row,
)
from red_team.log.memory.retriever import Embedder, make_default_embedder
from red_team.log.memory.store import Store

__all__ = [
    "MemoryRuntime",
    "MemoryHookConfig",
    "create_memory_runtime",
]


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryHookConfig:
    memory_mode: str = "off"            # off / static / evolving
    extractor_model: str = "deepseek-v4-pro"
    evolver_model: str = "deepseek-v4-pro"
    extractor_version: str = DEFAULT_EXTRACTOR_VERSION
    evolve_config: EvolveConfig = EvolveConfig()
    embedder_backend: str = "bge_small"  # plan default
    # Active campaign identity — frozen for the run.  Used by
    # retriever / evolver to enforce single-(seed, target) scope when
    # the prompt callers do not have them in scope.
    seed_id: str = ""
    target_model: str = ""
    # When set, the runtime writes to / reads from the shared sqlite
    # at this path (cross-seed memory pool, Phase D 5-dim).  When
    # unset, the runtime uses the per-run ``memory_db`` (single-seed
    # scope, Medium-plan default).
    shared_memory_db: str = ""
    # When True, retriever drops the source_seed_id filter so units
    # from other seeds are also returned (paired with shared_memory_db).
    cross_seed_retrieval: bool = False
    # Snapshot boundary for phase-2 retrieval — restricts retrieval
    # to ``rowid <= read_snapshot_rowid``.  ``0`` (default) disables.
    read_snapshot_rowid: int = 0
    # Optional environment overrides; defaults match the rest of the
    # red-team pipeline.
    api_base_env: str = "OPENAI_API_BASE"
    api_key_env: str = "OPENAI_API_KEY"
    request_timeout_env: str = "LLM_REQUEST_TIMEOUT"


@dataclass(slots=True)
class MemoryRuntime:
    """Lazy-initialised LLM client + Store + embedder shared across hook calls."""

    cfg: MemoryHookConfig
    memory_db: Path
    _client: Any = None
    _store: Store | None = None
    _embedder: Embedder | None = None

    @property
    def enabled(self) -> bool:
        return self.cfg.memory_mode == "evolving"

    @property
    def store(self) -> Store:
        if self._store is None:
            # Prefer the shared cross-seed sqlite when configured;
            # fall back to the per-run memory_db.
            db_path = (
                Path(self.cfg.shared_memory_db)
                if self.cfg.shared_memory_db
                else self.memory_db
            )
            self._store = Store(db_path)
        return self._store

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = make_default_embedder(
                backend=cast(Literal["bge_small", "jaccard"], self.cfg.embedder_backend)
            )
        return self._embedder

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            base = os.environ.get(self.cfg.api_base_env, "http://localhost:4000")
            key = os.environ.get(self.cfg.api_key_env, "sk-claw-local")
            try:
                timeout = float(os.environ.get(self.cfg.request_timeout_env, "300"))
            except ValueError:
                timeout = 300.0
            self._client = OpenAI(
                base_url=base, api_key=key, timeout=timeout, max_retries=0
            )
        return self._client

    # -------------------------------------------------------------- #
    # Hook entry point                                               #
    # -------------------------------------------------------------- #

    def on_harvest(self, row: HarvestRow) -> dict[str, Any]:
        """Extract → upsert → evolve.  Fail-safe: never raises."""
        if not self.enabled:
            return {"skipped": "memory_mode != evolving"}
        if row.refused:
            return {"skipped": "row.refused"}
        try:
            unit = extract_row(
                row,
                client=self.client,
                model=self.cfg.extractor_model,
                extractor_version=self.cfg.extractor_version,
            )
        except ExtractionError as exc:
            _LOGGER.warning(
                "memory.extract.fail  run_id=%s round=%d reason=%s",
                row.run_id, row.round, exc,
            )
            return {"extract_error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.exception("memory.extract.crash")
            return {"extract_crash": str(exc)}

        try:
            self.store.upsert_unit(unit)
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.exception("memory.upsert.crash")
            return {"upsert_crash": str(exc), "unit_id": unit.unit_id}

        evolve_result: dict[str, Any]
        # Mirror cross_seed flag so evolver searches across all seeds
        # when the runtime is in shared-memory mode.
        evolve_cfg = self.cfg.evolve_config
        if self.cfg.cross_seed_retrieval and not evolve_cfg.cross_seed:
            evolve_cfg = EvolveConfig(
                max_neighbors=evolve_cfg.max_neighbors,
                sim_threshold=evolve_cfg.sim_threshold,
                timeout_s=evolve_cfg.timeout_s,
                temperature=evolve_cfg.temperature,
                max_attempts=evolve_cfg.max_attempts,
                cross_seed=True,
            )
        try:
            res = evolve(
                unit,
                store=self.store,
                embedder=self.embedder,
                client=self.client,
                model=self.cfg.evolver_model,
                config=evolve_cfg,
            )
            evolve_result = {
                "triggered": res.triggered,
                "neighbors_considered": res.neighbors_considered,
                "units_evolved": list(res.units_evolved),
                "new_unit_ids": list(res.new_unit_ids),
                "fail_reason": res.fail_reason,
                "notes": list(res.notes),
            }
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.exception("memory.evolve.crash")
            evolve_result = {"crash": str(exc)}

        _LOGGER.info(
            "memory.hook  run_id=%s round=%d unit=%s category=%s evolve=%s",
            row.run_id, row.round, unit.unit_id[:12], unit.category,
            "ok" if evolve_result.get("triggered") else "skip",
        )
        return {
            "unit_id": unit.unit_id,
            "category": unit.category,
            "evolve": evolve_result,
        }


def create_memory_runtime(
    *,
    memory_db: Path | str,
    run_config: RunConfig,
    seed_id: str = "",
    target_model: str = "",
) -> MemoryRuntime:
    """Build a :class:`MemoryRuntime` from a :class:`RunConfig`."""
    cfg = MemoryHookConfig(
        memory_mode=run_config.memory_mode,
        extractor_model=run_config.memory_extractor_model,
        evolver_model=run_config.memory_evolver_model,
        seed_id=seed_id,
        target_model=target_model,
        shared_memory_db=run_config.shared_memory_db,
        cross_seed_retrieval=run_config.cross_seed_retrieval,
        read_snapshot_rowid=run_config.read_snapshot_rowid,
    )
    return MemoryRuntime(cfg=cfg, memory_db=Path(memory_db))
