"""Tests for ``red_team.log.memory.store`` and ``bootstrap``."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from core.schemas import AttackObjective, Channel, Topology

from red_team.log.memory.bootstrap import load_yaml_seeds
from red_team.log.memory.store import Store
from red_team.log.memory.unit import MemoryUnit


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unit(
    unit_id: str = "u1",
    *,
    technique: str = "module-import-write",
    category: str = "positive",
    seed_id: str = "seed-04",
    target_model: str = "gpt-5.4-mini",
    bypass: bool = True,
    lethal: bool = True,
    evolution_step: int = 0,
    superseded_by: str | None = None,
) -> MemoryUnit:
    return MemoryUnit(
        unit_id=unit_id,
        source="harvested",
        source_run_id="run-1",
        source_seed_id=seed_id,
        variant_hash="v0",
        extractor_version="v0.1.0-dev",
        authored_by="test",
        created_at=_now(),
        evolution_step=evolution_step,
        superseded_by=superseded_by,
        o=AttackObjective.PRIVILEGE_ESCALATION,
        tau=Topology.SINGLE,
        kappa=(Channel.FILESYSTEM,),
        chain=("slopbuster",),
        technique_name=technique,
        mechanism="Plant decoy and chmod 4755 it.",
        exploited_assumption="Vetter only matches explicit setuid syntax.",
        preconditions=("/tmp writable",),
        tags=("module-load", "import-time"),
        bypass=bypass,
        lethal=lethal,
        target_model=target_model,
        blue_reason_codes=(),
        category=category,
    )


def test_store_init_creates_schema(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    assert store.count_units() == 0
    assert store.count_active() == 0


def test_upsert_and_query_latest(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    u = _unit("u1")
    store.upsert_unit(u)
    hits = store.query_latest(
        source_seed_id="seed-04", target_model="gpt-5.4-mini"
    )
    assert len(hits) == 1 and hits[0].unit_id == "u1"

    # Upsert with same id replaces in place.
    store.upsert_unit(_unit("u1", technique="renamed"))
    assert store.count_units() == 1
    assert store.query_latest(
        source_seed_id="seed-04", target_model="gpt-5.4-mini"
    )[0].technique_name == "renamed"


def test_query_latest_seed_target_joint_key(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    store.upsert_unit(_unit("u_x", target_model="gpt-5.4-mini"))
    store.upsert_unit(_unit("u_y", target_model="kimi-2.5"))
    only_x = store.query_latest(
        source_seed_id="seed-04", target_model="gpt-5.4-mini"
    )
    assert {h.unit_id for h in only_x} == {"u_x"}
    only_y = store.query_latest(
        source_seed_id="seed-04", target_model="kimi-2.5"
    )
    assert {h.unit_id for h in only_y} == {"u_y"}


def test_query_latest_excludes_superseded(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    store.upsert_unit(_unit("u_v0"))
    store.upsert_unit(_unit("u_v1", evolution_step=1))
    store.set_superseded("u_v0", by="u_v1")
    latest = store.query_latest(
        source_seed_id="seed-04", target_model="gpt-5.4-mini"
    )
    assert {h.unit_id for h in latest} == {"u_v1"}
    # but the chain still walkable
    chain = store.evolution_chain("u_v0")
    assert [u.unit_id for u in chain] == ["u_v0", "u_v1"]


def test_query_by_category(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    store.commit_batch([
        _unit("p1", category="positive"),
        _unit("c1", category="chokepoint", bypass=False, lethal=False),
        _unit("c2", category="chokepoint", bypass=True, lethal=False),
    ])
    pos = store.query_by_category(
        source_seed_id="seed-04", target_model="gpt-5.4-mini",
        category="positive",
    )
    chokes = store.query_by_category(
        source_seed_id="seed-04", target_model="gpt-5.4-mini",
        category="chokepoint",
    )
    assert {h.unit_id for h in pos} == {"p1"}
    assert {h.unit_id for h in chokes} == {"c1", "c2"}


def test_embedding_cache(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    store.upsert_unit(_unit("u1"))
    assert store.get_embedding("u1") is None
    store.set_embedding("u1", b"\x01\x02\x03\x04")
    assert store.get_embedding("u1") == b"\x01\x02\x03\x04"


def test_meta_persisted(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    store.set_meta("current_extractor_version", "v0.1.0-dev")
    assert store.get_meta("current_extractor_version") == "v0.1.0-dev"


def test_count_by_evolution_step(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    store.commit_batch([
        _unit("a", evolution_step=0),
        _unit("b", evolution_step=0),
        _unit("c", evolution_step=1),
    ])
    counts = store.count_by_evolution_step()
    assert counts == {0: 2, 1: 1}


def test_bootstrap_loads_yaml_seeds(tmp_path: Path):
    seeds_root = tmp_path / "data" / "memory" / "seeds"
    pos_dir = seeds_root / "pos"
    pos_dir.mkdir(parents=True)
    (pos_dir / "ex.yaml").write_text(
        yaml.safe_dump({
            "unit_id": "yamlseed.ex.v1",
            "source": "seed",
            "category": "positive",
            "o": "privilege_escalation",
            "tau": "single",
            "kappa": ["filesystem"],
            "chain": ["slopbuster"],
            "technique_name": "yaml-seed-test",
            "mechanism": "test mechanism",
            "exploited_assumption": "test assumption",
            "preconditions": ["test precondition"],
            "tags": ["benign"],
            "bypass": True,
            "lethal": True,
            "target_model": "gpt-5.4-mini",
            "extractor_version": "v0.1.0-dev",
            "authored_by": "test",
            "created_at": _now(),
            "evolution_step": 0,
            "superseded_by": None,
        }),
        encoding="utf-8",
    )
    store = Store(tmp_path / "m.sqlite")
    lock = tmp_path / "data" / "memory" / "seeds.lock"
    committed, skipped = load_yaml_seeds(
        store, seeds_root=seeds_root, lock_path=lock
    )
    assert (committed, skipped) == (1, 0)
    assert store.count_units() == 1
    # Idempotent: rerun must not double-commit.
    committed2, skipped2 = load_yaml_seeds(
        store, seeds_root=seeds_root, lock_path=lock
    )
    assert (committed2, skipped2) == (0, 1)
    assert store.count_units() == 1


def test_bootstrap_real_example_seed_yaml(tmp_path: Path):
    """The repo's real data/memory/seeds/pos/example_seed.yaml must load."""
    repo_seeds = Path("data/memory/seeds")
    if not (repo_seeds / "pos" / "example_seed.yaml").exists():
        pytest.skip("example_seed.yaml not present in this checkout")
    store = Store(tmp_path / "m.sqlite")
    lock = tmp_path / "seeds.lock"
    committed, _ = load_yaml_seeds(
        store, seeds_root=repo_seeds, lock_path=lock
    )
    assert committed >= 1
