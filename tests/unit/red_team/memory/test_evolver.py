"""Tests for ``red_team.log.memory.evolver``.

Covers user-mandated invariants (plan §extra-2.4):

* Fail-safe: LLM timeout / parse fail / empty / unknown unit_id all
  return without raising.
* Hard step cap (``MAX_EVOLUTION_STEPS = 3``) is honoured even when
  the LLM tries to push past it.
* Facts (technique_name / bypass / lethal / blue_reason_codes / o /
  tau / kappa / chain / target_model) are carried verbatim through
  ``with_evolution``; the LLM cannot rewrite them.
* ``superseded_by`` is set on the old unit; the new unit
  has ``superseded_by IS NULL`` and shows up in ``query_latest``.
* Single-(seed, target) scope: neighbours from a different
  ``target_model`` cell are never even considered.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.schemas import AttackObjective, Channel, Topology

from red_team.log.memory.evolver import (
    EVOLVE_PROMPT,
    EvolveConfig,
    MAX_EVOLUTION_STEPS,
    evolve,
)
from red_team.log.memory.retriever import JaccardEmbedder
from red_team.log.memory.store import Store
from red_team.log.memory.unit import MemoryUnit


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _u(
    *,
    unit_id: str,
    seed: str = "seed-04",
    target: str = "gpt-5.4-mini",
    category: str = "positive",
    technique: str = "module-import-write",
    mechanism: str = "Hide a Python open() in __init__.py to overwrite a sentinel before agent action",
    tags: tuple[str, ...] = ("module-load-side-effect", "import-time-write"),
    bypass: bool = True,
    lethal: bool = True,
    evolution_step: int = 0,
    superseded_by: str | None = None,
    blue_reason_codes: tuple[str, ...] = ("FACT_REASON",),
) -> MemoryUnit:
    return MemoryUnit(
        unit_id=unit_id, source="harvested",
        source_run_id="r1", source_seed_id=seed,
        variant_hash="v" + unit_id, extractor_version="v0",
        authored_by="t", created_at=_now(),
        evolution_step=evolution_step, superseded_by=superseded_by,
        o=AttackObjective.PRIVILEGE_ESCALATION,
        tau=Topology.SINGLE,
        kappa=(Channel.FILESYSTEM,),
        chain=("c",),
        technique_name=technique,
        mechanism=mechanism,
        exploited_assumption="vetter inspects only SKILL.md, not entry_script body",
        preconditions=("__init__.py importable",),
        tags=tags,
        bypass=bypass, lethal=lethal, target_model=target,
        blue_reason_codes=blue_reason_codes,
        category=category,
    )


# ----- Fake LLM clients -----


class _ScriptedClient:
    """Minimal client.chat.completions.create stub returning a scripted text."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls = 0

    def with_options(self, **_kw):
        return self

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **_kw):
        self.calls += 1
        return _Response(self._response_text)


class _RaisingClient:
    """Client that raises on every call (simulates timeout)."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def with_options(self, **_kw):
        return self

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **_kw):
        raise self._exc


class _Response:
    def __init__(self, text: str) -> None:
        self.choices = [_Choice(text)]


class _Choice:
    def __init__(self, text: str) -> None:
        self.message = _Message(text)


class _Message:
    def __init__(self, text: str) -> None:
        self.content = text


# ----- Tests -----


def test_evolve_no_neighbors_returns_safely(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    new = _u(unit_id="new")
    store.upsert_unit(new)
    res = evolve(
        new, store=store, embedder=JaccardEmbedder(),
        client=_ScriptedClient('{"evolved": []}'),
    )
    assert res.triggered is False
    assert res.units_evolved == ()
    assert "no eligible neighbours" in " | ".join(res.notes)


def test_evolve_no_family_detected_is_logged_not_failed(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    # Two related units so neighbours pass the embedding threshold.
    a = _u(unit_id="a")
    b = _u(unit_id="b")
    store.upsert_unit(a)
    store.upsert_unit(b)
    res = evolve(
        a, store=store, embedder=JaccardEmbedder(),
        client=_ScriptedClient('{"evolved": []}'),
        config=EvolveConfig(sim_threshold=0.0),  # accept any cosine
    )
    assert res.triggered is True
    assert res.fail_reason is None
    assert "no_family_detected" in " | ".join(res.notes)
    assert store.count_units() == 2  # nothing new written


def test_evolve_writes_successor_and_supersedes_old(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    a = _u(unit_id="a")
    b = _u(unit_id="b")
    store.upsert_unit(a)
    store.upsert_unit(b)

    response = json.dumps({
        "evolved": [
            {
                "unit_id": "b",
                "mechanism": "evolved family-level mechanism for module-load attacks in general",
                "exploited_assumption": "vetter blind to import-time runtime side effects across all entry-script naming conventions",
                "preconditions": ["import-time entrypoint exists"],
                "tags": ["module-load-family", "import-time-side-effect"],
            },
        ],
    })
    res = evolve(
        a, store=store, embedder=JaccardEmbedder(),
        client=_ScriptedClient(response),
        config=EvolveConfig(sim_threshold=0.0),
    )
    assert res.triggered is True
    assert res.units_evolved == ("b",)
    assert len(res.new_unit_ids) == 1

    # b is now superseded; query_latest only returns a + the new b' (not the old b).
    latest = store.query_latest(source_seed_id="seed-04", target_model="gpt-5.4-mini")
    ids = {u.unit_id for u in latest}
    assert "b" not in ids, "old b must be hidden by superseded_by filter"
    assert "a" in ids
    assert res.new_unit_ids[0] in ids
    # The successor has step=1 and evolved-style tags.
    successor = store.get_unit(res.new_unit_ids[0])
    assert successor.evolution_step == 1
    assert "module-load-family" in successor.tags


def test_evolve_facts_never_change(tmp_path: Path):
    """LLM tries to rewrite facts; evolver must ignore those keys."""
    store = Store(tmp_path / "m.sqlite")
    a = _u(unit_id="a", bypass=True, lethal=True, technique="real-technique-name",
           blue_reason_codes=("REAL_REASON",))
    b = _u(unit_id="b", bypass=True, lethal=True, technique="real-technique-name",
           blue_reason_codes=("REAL_REASON",))
    store.upsert_unit(a)
    store.upsert_unit(b)
    response = json.dumps({
        "evolved": [
            {
                "unit_id": "b",
                "mechanism": "evolved mechanism describing module-load family in general terms",
                "exploited_assumption": "vetter blind to import-time runtime side effects in general",
                "preconditions": ["any import-time entrypoint"],
                "tags": ["family-level"],
                # Forbidden keys — must be ignored.
                "technique_name": "FAKE-NEW-TECHNIQUE",
                "bypass": False,
                "lethal": False,
                "blue_reason_codes": ["FAKE_REASON"],
            },
        ],
    })
    res = evolve(
        a, store=store, embedder=JaccardEmbedder(),
        client=_ScriptedClient(response),
        config=EvolveConfig(sim_threshold=0.0),
    )
    assert res.units_evolved == ("b",)
    assert any("rewrite facts" in n.lower() for n in res.notes)
    new_b = store.get_unit(res.new_unit_ids[0])
    # Facts carried verbatim from b.
    assert new_b.technique_name == "real-technique-name"
    assert new_b.bypass is True
    assert new_b.lethal is True
    assert new_b.blue_reason_codes == ("REAL_REASON",)


def test_evolve_step_cap_enforced(tmp_path: Path):
    """A unit at MAX_EVOLUTION_STEPS=3 is excluded from candidates."""
    store = Store(tmp_path / "m.sqlite")
    saturated = _u(unit_id="saturated", evolution_step=MAX_EVOLUTION_STEPS)
    new = _u(unit_id="new")
    store.upsert_unit(saturated)
    store.upsert_unit(new)
    response = json.dumps({
        "evolved": [
            # LLM tries to evolve the saturated unit anyway; but it
            # was never given as a candidate, so it gets ignored.
            {
                "unit_id": "saturated",
                "mechanism": "should-be-ignored mechanism not actually applied because unit excluded",
                "exploited_assumption": "should also be ignored",
                "preconditions": ["x"], "tags": ["x"],
            },
        ],
    })
    res = evolve(
        new, store=store, embedder=JaccardEmbedder(),
        client=_ScriptedClient(response),
        config=EvolveConfig(sim_threshold=0.0),
    )
    # No actual successor was written.
    assert res.units_evolved == ()
    # The saturated unit is still at step=3 with original mechanism.
    saturated_now = store.get_unit("saturated")
    assert saturated_now.evolution_step == MAX_EVOLUTION_STEPS
    assert "ignored" not in saturated_now.mechanism


def test_evolve_seed_target_isolation(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    a = _u(unit_id="a", target="gpt-5.4-mini")
    b_other = _u(unit_id="b_other", target="kimi-2.5")
    store.upsert_unit(a)
    store.upsert_unit(b_other)
    res = evolve(
        a, store=store, embedder=JaccardEmbedder(),
        client=_ScriptedClient('{"evolved": []}'),
        config=EvolveConfig(sim_threshold=0.0),
    )
    # b_other is in a different (seed, target) cell → never considered.
    assert res.triggered is False


def test_evolve_timeout_does_not_crash(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    a = _u(unit_id="a")
    b = _u(unit_id="b")
    store.upsert_unit(a)
    store.upsert_unit(b)
    res = evolve(
        a, store=store, embedder=JaccardEmbedder(),
        client=_RaisingClient(TimeoutError("call timed out")),
        config=EvolveConfig(sim_threshold=0.0, timeout_s=5.0),
    )
    assert res.triggered is True
    assert res.fail_reason is not None
    assert "timeout" in res.fail_reason
    # Run continues; nothing was supersedded.
    assert store.count_units() == 2


def test_evolve_parse_fail_does_not_crash(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    a = _u(unit_id="a")
    b = _u(unit_id="b")
    store.upsert_unit(a)
    store.upsert_unit(b)
    res = evolve(
        a, store=store, embedder=JaccardEmbedder(),
        # Two consecutive non-JSON responses → parse_fail.
        client=_ScriptedClient("not even close to json {garbage"),
        config=EvolveConfig(sim_threshold=0.0, max_attempts=2),
    )
    assert res.triggered is True
    assert res.fail_reason is not None
    assert "parse_fail" in res.fail_reason


def test_evolve_unknown_unit_id_logged_and_skipped(tmp_path: Path):
    store = Store(tmp_path / "m.sqlite")
    a = _u(unit_id="a")
    b = _u(unit_id="b")
    store.upsert_unit(a)
    store.upsert_unit(b)
    response = json.dumps({
        "evolved": [
            {
                "unit_id": "ghost",  # not in candidates
                "mechanism": "x" * 30, "exploited_assumption": "y" * 20,
                "preconditions": ["z" * 5], "tags": ["t"],
            },
        ],
    })
    res = evolve(
        a, store=store, embedder=JaccardEmbedder(),
        client=_ScriptedClient(response),
        config=EvolveConfig(sim_threshold=0.0),
    )
    assert res.units_evolved == ()
    assert any("unknown unit_id" in n for n in res.notes)


def test_evolve_prompt_template_contains_fact_constraints():
    # Defensive: if someone edits the prompt and accidentally drops the
    # fact-immutability instruction, this test catches it.
    assert "Do NOT rewrite facts" in EVOLVE_PROMPT
    assert "technique_name" in EVOLVE_PROMPT
    assert "bypass" in EVOLVE_PROMPT
    assert "lethal" in EVOLVE_PROMPT
