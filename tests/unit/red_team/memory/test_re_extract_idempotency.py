"""IMPL-07 idempotency tests.

Hits the deterministic ``unit_id`` property without making any LLM
call by constructing MemoryUnit objects directly and writing them to
a Store.  Two invariants:

1. Same ``extractor_version`` + same ``(variant_hash, run_id, round)``
   ⇒ same ``unit_id`` ⇒ Store keeps row count unchanged on re-upsert.
2. Different ``extractor_version`` ⇒ different ``unit_id`` ⇒ both
   units coexist in the same store.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from core.schemas import AttackObjective, Channel, Topology

from red_team.log.memory.extractor import build_unit_id
from red_team.log.memory.store import Store
from red_team.log.memory.unit import MemoryUnit


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_unit(*, version: str, variant_hash: str = "vh-x") -> MemoryUnit:
    uid = build_unit_id(
        variant_hash=variant_hash, run_id="run-1", round_index=2,
        extractor_version=version,
    )
    return MemoryUnit(
        unit_id=uid,
        source="harvested",
        source_run_id="run-1",
        source_seed_id="seed-04",
        variant_hash=variant_hash,
        extractor_version=version,
        authored_by=f"extractor:test:{version}",
        created_at=_now(),
        evolution_step=0, superseded_by=None,
        o=AttackObjective.PRIVILEGE_ESCALATION,
        tau=Topology.SINGLE,
        kappa=(Channel.FILESYSTEM,),
        chain=("slopbuster",),
        technique_name="t",
        mechanism="m" * 30,
        exploited_assumption="ea" * 11,
        preconditions=("p1",),
        tags=("tag-a",),
        bypass=True, lethal=True,
        target_model="gpt-5.4-mini",
        blue_reason_codes=(),
        category="positive",
    )


def test_unit_id_deterministic():
    a = build_unit_id(variant_hash="x", run_id="r", round_index=1, extractor_version="v0")
    b = build_unit_id(variant_hash="x", run_id="r", round_index=1, extractor_version="v0")
    assert a == b


def test_unit_id_changes_with_version():
    a = build_unit_id(variant_hash="x", run_id="r", round_index=1, extractor_version="v0")
    b = build_unit_id(variant_hash="x", run_id="r", round_index=1, extractor_version="v1")
    assert a != b


def test_same_version_reupsert_idempotent(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    u = _build_unit(version="v0.1.0")
    store.upsert_unit(u)
    store.upsert_unit(u)  # second call must not double-row
    assert store.count_units() == 1


def test_different_versions_coexist(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    u_v0 = _build_unit(version="v0.1.0")
    u_v1 = _build_unit(version="v1.0.0")
    store.upsert_unit(u_v0)
    store.upsert_unit(u_v1)
    assert store.count_units() == 2
    # Both versions persist; the run-scope query_latest sees both
    # because neither has been superseded.
    latest = store.query_latest(
        source_seed_id="seed-04", target_model="gpt-5.4-mini"
    )
    assert {h.unit_id for h in latest} == {u_v0.unit_id, u_v1.unit_id}
