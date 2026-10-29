"""Per-run SQLite store for :class:`MemoryUnit` rows.

Medium-scope (plan §5.3): one ``memory.sqlite`` per phase-1 run, single
``memory_units`` table with a ``superseded_by``-based evolution chain
plus a cached ``embedding`` BLOB column.  No FTS5, no vss0, no edge
table — the Medium plan simplified all of that away.

Read paths the offline pipeline + retriever rely on:

* :meth:`query_latest`        — latest version (``superseded_by IS NULL``)
                                 strictly filtered by
                                 ``(source_seed_id, target_model)``.
* :meth:`query_by_category`   — same filter, optional category.
* :meth:`get_unit`            — one unit by ``unit_id`` regardless of
                                 evolution lineage (used by evolver to
                                 link new ↔ old).

Write paths:

* :meth:`upsert_unit`         — insert-or-replace one unit (used by
                                 extractor and evolver).
* :meth:`commit_batch`        — bulk insert/replace.
* :meth:`set_superseded`      — flip the ``superseded_by`` pointer on
                                 an old unit when the evolver writes
                                 its successor.
* :meth:`set_embedding`       — cache a numpy float32 vector against a
                                 ``unit_id`` (called once at first
                                 upsert; never overwritten).

The ``embedding`` cache prevents the evolver from re-embedding every
neighbour on every round (plan §extra-2.3 — embedding cache is
mandatory not optional).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from core.schemas.attack_objective import AttackObjective, Channel, Topology
from red_team.log.memory.unit import MemoryUnit

__all__ = ["Store"]


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_units (
    unit_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    evolution_step INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT,
    o TEXT,
    tau TEXT,
    kappa_json TEXT,
    chain_json TEXT,
    technique_name TEXT,
    mechanism TEXT,
    exploited_assumption TEXT,
    preconditions_json TEXT,
    tags_json TEXT,
    bypass INTEGER NOT NULL,
    lethal INTEGER NOT NULL,
    target_model TEXT,
    blue_reason_codes_json TEXT,
    variant_hash TEXT,
    source_run_id TEXT,
    source_seed_id TEXT,
    authored_by TEXT,
    created_at TEXT,
    embedding BLOB
);

-- Single most-used query path: latest units in a (seed, target) cell.
CREATE INDEX IF NOT EXISTS idx_units_seed_target_active
    ON memory_units(source_seed_id, target_model, superseded_by);

CREATE INDEX IF NOT EXISTS idx_units_category
    ON memory_units(source_seed_id, target_model, category, superseded_by);

CREATE TABLE IF NOT EXISTS memory_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class Store:
    """SQLite wrapper for one run's memory."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            # WAL mode lets multiple writers from different processes
            # serialize cleanly without locking errors — required for
            # the Phase D 5-dim coordinator where 4 seed runners share
            # one sqlite.  WAL persists across reopen, so set it at
            # init.
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=30000")  # 30s wait on locks
            con.executescript(_SCHEMA_SQL)
            con.commit()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        # 30s busy_timeout for cross-process contention; WAL mode is
        # set at __init__ and persists.
        con = sqlite3.connect(str(self.path), timeout=30.0)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    # ------------------------------------------------------------------ #
    # Write                                                              #
    # ------------------------------------------------------------------ #

    def upsert_unit(self, unit: MemoryUnit) -> None:
        self.commit_batch([unit])

    def commit_batch(self, units: Iterable[MemoryUnit]) -> None:
        rows = [_unit_to_row(u) for u in units]
        if not rows:
            return
        with self._conn() as con:
            con.executemany(_UPSERT_UNIT_SQL, rows)
            con.commit()

    def set_superseded(self, old_unit_id: str, *, by: str) -> None:
        """Flip the ``superseded_by`` pointer on ``old_unit_id``.

        Called by the evolver after writing the new (step+1) version.
        Idempotent — re-running with the same ``by`` is a no-op.
        """
        with self._conn() as con:
            con.execute(
                "UPDATE memory_units SET superseded_by = ? WHERE unit_id = ?",
                (by, old_unit_id),
            )
            con.commit()

    def set_embedding(self, unit_id: str, vec: bytes) -> None:
        """Store a precomputed embedding for one unit.

        ``vec`` is the bytes of a float32 numpy array (caller's
        responsibility to encode).  Idempotent: overwrites prior value.
        """
        with self._conn() as con:
            con.execute(
                "UPDATE memory_units SET embedding = ? WHERE unit_id = ?",
                (vec, unit_id),
            )
            con.commit()

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO memory_meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            con.commit()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT value FROM memory_meta WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    # ------------------------------------------------------------------ #
    # Read                                                               #
    # ------------------------------------------------------------------ #

    def get_unit(self, unit_id: str) -> MemoryUnit | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM memory_units WHERE unit_id = ?", (unit_id,)
            ).fetchone()
            return _row_to_unit(row) if row else None

    def get_embedding(self, unit_id: str) -> bytes | None:
        with self._conn() as con:
            row = con.execute(
                "SELECT embedding FROM memory_units WHERE unit_id = ?",
                (unit_id,),
            ).fetchone()
            if not row or row["embedding"] is None:
                return None
            return bytes(row["embedding"])

    def query_latest(
        self,
        *,
        source_seed_id: str | None,
        target_model: str | None,
        limit: int | None = None,
        max_rowid: int | None = None,
    ) -> list[MemoryUnit]:
        """Latest (``superseded_by IS NULL``) units, optionally filtered.

        ``source_seed_id=None`` and / or ``target_model=None`` relax
        the corresponding filter — used by the cross-seed shared-memory
        coordinator (Phase D 5-dim).  ``max_rowid`` further restricts
        results to rows visible at a given snapshot boundary (used by
        phase-2 retrieval to avoid seeing same-batch sibling writes).
        """
        clauses = ["superseded_by IS NULL"]
        params: list[Any] = []
        if source_seed_id is not None:
            clauses.append("source_seed_id = ?")
            params.append(source_seed_id)
        if target_model is not None:
            clauses.append("target_model = ?")
            params.append(target_model)
        if max_rowid is not None:
            clauses.append("rowid <= ?")
            params.append(int(max_rowid))
        sql = (
            "SELECT * FROM memory_units WHERE "
            + " AND ".join(clauses)
            + " ORDER BY evolution_step DESC, created_at DESC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._conn() as con:
            return [_row_to_unit(r) for r in con.execute(sql, params).fetchall()]

    def query_by_category(
        self,
        *,
        source_seed_id: str | None,
        target_model: str | None,
        category: str,
        limit: int | None = None,
        max_rowid: int | None = None,
    ) -> list[MemoryUnit]:
        clauses = ["category = ?", "superseded_by IS NULL"]
        params: list[Any] = [category]
        if source_seed_id is not None:
            clauses.append("source_seed_id = ?")
            params.append(source_seed_id)
        if target_model is not None:
            clauses.append("target_model = ?")
            params.append(target_model)
        if max_rowid is not None:
            clauses.append("rowid <= ?")
            params.append(int(max_rowid))
        sql = (
            "SELECT * FROM memory_units WHERE "
            + " AND ".join(clauses)
            + " ORDER BY evolution_step DESC, created_at DESC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._conn() as con:
            return [_row_to_unit(r) for r in con.execute(sql, params).fetchall()]

    def current_max_rowid(self) -> int:
        """Return the largest ``rowid`` currently in ``memory_units``.

        Used by the phase-2 coordinator to record a snapshot boundary
        — phase-2 retrievers query with ``max_rowid=<this value>`` so
        they never see same-batch sibling writes.
        """
        with self._conn() as con:
            row = con.execute(
                "SELECT MAX(rowid) AS m FROM memory_units"
            ).fetchone()
            return int(row["m"] or 0)

    def evolution_chain(self, unit_id: str) -> list[MemoryUnit]:
        """Walk the ``superseded_by`` chain starting at ``unit_id``."""
        out: list[MemoryUnit] = []
        cur_id: str | None = unit_id
        while cur_id is not None:
            u = self.get_unit(cur_id)
            if u is None:
                break
            out.append(u)
            cur_id = u.superseded_by
        return out

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def count_units(self) -> int:
        with self._conn() as con:
            return int(con.execute("SELECT COUNT(*) AS n FROM memory_units").fetchone()["n"])

    def count_active(self) -> int:
        with self._conn() as con:
            return int(
                con.execute(
                    "SELECT COUNT(*) AS n FROM memory_units WHERE superseded_by IS NULL"
                ).fetchone()["n"]
            )

    def count_by_evolution_step(self) -> dict[int, int]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT evolution_step AS s, COUNT(*) AS n "
                "FROM memory_units GROUP BY evolution_step ORDER BY s"
            ).fetchall()
            return {int(r["s"]): int(r["n"]) for r in rows}


# ---------------------------------------------------------------------------
# Row encoding helpers
# ---------------------------------------------------------------------------


_UPSERT_UNIT_SQL = """
INSERT INTO memory_units (
    unit_id, source, category, extractor_version,
    evolution_step, superseded_by,
    o, tau, kappa_json, chain_json,
    technique_name, mechanism, exploited_assumption,
    preconditions_json, tags_json,
    bypass, lethal, target_model,
    blue_reason_codes_json,
    variant_hash, source_run_id, source_seed_id,
    authored_by, created_at
) VALUES (
    :unit_id, :source, :category, :extractor_version,
    :evolution_step, :superseded_by,
    :o, :tau, :kappa_json, :chain_json,
    :technique_name, :mechanism, :exploited_assumption,
    :preconditions_json, :tags_json,
    :bypass, :lethal, :target_model,
    :blue_reason_codes_json,
    :variant_hash, :source_run_id, :source_seed_id,
    :authored_by, :created_at
)
ON CONFLICT(unit_id) DO UPDATE SET
    source = excluded.source,
    category = excluded.category,
    extractor_version = excluded.extractor_version,
    evolution_step = excluded.evolution_step,
    superseded_by = excluded.superseded_by,
    o = excluded.o, tau = excluded.tau,
    kappa_json = excluded.kappa_json, chain_json = excluded.chain_json,
    technique_name = excluded.technique_name,
    mechanism = excluded.mechanism,
    exploited_assumption = excluded.exploited_assumption,
    preconditions_json = excluded.preconditions_json,
    tags_json = excluded.tags_json,
    bypass = excluded.bypass, lethal = excluded.lethal,
    target_model = excluded.target_model,
    blue_reason_codes_json = excluded.blue_reason_codes_json,
    variant_hash = excluded.variant_hash,
    source_run_id = excluded.source_run_id,
    source_seed_id = excluded.source_seed_id,
    authored_by = excluded.authored_by,
    created_at = excluded.created_at;
"""


def _unit_to_row(u: MemoryUnit) -> dict[str, Any]:
    return {
        "unit_id": u.unit_id,
        "source": u.source,
        "category": u.category,
        "extractor_version": u.extractor_version,
        "evolution_step": int(u.evolution_step),
        "superseded_by": u.superseded_by,
        "o": u.o.value, "tau": u.tau.value,
        "kappa_json": json.dumps([c.value for c in u.kappa]),
        "chain_json": json.dumps(list(u.chain)),
        "technique_name": u.technique_name,
        "mechanism": u.mechanism,
        "exploited_assumption": u.exploited_assumption,
        "preconditions_json": json.dumps(list(u.preconditions)),
        "tags_json": json.dumps(list(u.tags)),
        "bypass": int(bool(u.bypass)),
        "lethal": int(bool(u.lethal)),
        "target_model": u.target_model,
        "blue_reason_codes_json": json.dumps(list(u.blue_reason_codes)),
        "variant_hash": u.variant_hash,
        "source_run_id": u.source_run_id,
        "source_seed_id": u.source_seed_id,
        "authored_by": u.authored_by,
        "created_at": u.created_at,
    }


def _row_to_unit(r: sqlite3.Row) -> MemoryUnit:
    return MemoryUnit(
        unit_id=r["unit_id"],
        source=r["source"],
        source_run_id=r["source_run_id"],
        source_seed_id=r["source_seed_id"],
        variant_hash=r["variant_hash"],
        extractor_version=r["extractor_version"],
        authored_by=r["authored_by"] or "",
        created_at=r["created_at"] or "",
        evolution_step=int(r["evolution_step"] or 0),
        superseded_by=r["superseded_by"],
        o=AttackObjective(r["o"]),
        tau=Topology(r["tau"]),
        kappa=tuple(Channel(v) for v in json.loads(r["kappa_json"] or "[]")),
        chain=tuple(json.loads(r["chain_json"] or "[]")),
        technique_name=r["technique_name"] or "",
        mechanism=r["mechanism"] or "",
        exploited_assumption=r["exploited_assumption"] or "",
        preconditions=tuple(json.loads(r["preconditions_json"] or "[]")),
        tags=tuple(json.loads(r["tags_json"] or "[]")),
        bypass=bool(r["bypass"]),
        lethal=bool(r["lethal"]),
        target_model=r["target_model"] or "",
        blue_reason_codes=tuple(json.loads(r["blue_reason_codes_json"] or "[]")),
        category=r["category"],
    )
