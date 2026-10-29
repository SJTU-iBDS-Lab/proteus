"""Tests for :class:`red_team.campaign.runner._PerEntryHandler`.

Covers the per-entry checkpoint handler used during S2/S3 expansion:

* Every entry passed to the callback gets ``bypassed`` overwritten
  with the blue-team verdict (was ``None`` on input).
* Same-content mutations share one audit call via the internal
  audit cache (keyed on ``content_hash()``).
* ``finalize()`` returns ``(audits_issued, cache_hits)`` so the caller
  can record per-seed telemetry.
* ``blue_team_fn`` seam still routes correctly when supplied.
* The driver — not this helper — is responsible for recomputing
  ``funnel.succeeded``; that lives in the call sites in
  ``runner.py`` so we test it via the integration not here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import (
    AttackObjective,
    Channel,
    Evidence,
    PaperFinding,
    SkillVariant,
    Topology,
)
from red_team.campaign import runner as runner_mod
from red_team.evolution.path_expansion import PathExpandResult
from red_team.evolution.surface_expansion import SurfaceResult
from sandbox.honeypot import HoneypotResult


def _variant(tag: str) -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=[Channel.FILESYSTEM],
        d=f"doc::{tag}",
        chain=["skill-alpha"],
        metadata={"tag": tag},
    )


def _path_row(variant: SkillVariant, *, lethal: bool, score: float) -> PathExpandResult:
    return PathExpandResult(
        mutation=variant,
        tool_chain=[],
        hp=HoneypotResult(files_created=[]),
        bypassed=None,  # A1 contract: path_expand emits None
        lethal=lethal,
        family_candidate={"name": variant.metadata.get("tag", "?")},
        score=score,
        rationale="",
    )


def _surface_row(
    variant: SkillVariant, *, lethal: bool, post_hoc_ok: bool
) -> SurfaceResult:
    return SurfaceResult(
        mutation=variant,
        hp=HoneypotResult(files_created=[]),
        bypassed=None,  # A1 contract: surface_expand emits None
        lethal=lethal,
        o_prime=AttackObjective.DATA_EXFILTRATION,
        shared_motif={},
        post_hoc_ok=post_hoc_ok,
        rationale="",
    )


def _high_finding() -> PaperFinding:
    return PaperFinding(
        finding_id="x",
        layer="A",
        category="guidance_injection",
        reason_code="autonomy_encouragement",
        confidence=1.0,
        severity="high",
        evidence=Evidence(snippet=""),
        explanation="",
    )


def _make_handler(tmp_path, *, blue_team_fn=None):
    """Construct a per-entry handler with disk I/O routed at tmp_path."""
    parent = _variant("parent")
    handler = runner_mod._PerEntryHandler(
        repo=SimpleNamespace(),
        blue_team="ours",
        blue_team_fn=blue_team_fn,
        sandbox_container="ct",
        log_path=tmp_path / "log.jsonl",
        harvest_path=None,  # disable harvest write for these unit tests
        backend_tag="test",
        run_id="r1",
        seed_id="s1",
        parent_variant=parent,
        target_model="t",
        memory_runtime=None,
    )
    return handler.on_path_result, handler.on_surface_executed, handler.finalize


def test_per_entry_handler_overwrites_bypassed_on_every_entry(monkeypatch, tmp_path):
    """All entries leave the callback with bypassed ∈ {True, False}."""
    v_a, v_b, v_c = _variant("a"), _variant("b"), _variant("c")

    def fake_call_blue_team(variant, repo, method, fn, *, sandbox_container):
        return [_high_finding()] if variant.metadata["tag"] == "b" else []

    monkeypatch.setattr(runner_mod._seams, "call_blue_team", fake_call_blue_team)

    on_path, on_surface, finalize = _make_handler(tmp_path)
    rows = [
        _path_row(v_a, lethal=True, score=5.0),
        _path_row(v_b, lethal=True, score=4.0),
    ]
    surf = _surface_row(v_c, lethal=True, post_hoc_ok=True)
    for r in rows:
        on_path(r)
    on_surface(surf)

    assert rows[0].bypassed is True   # no high finding
    assert rows[1].bypassed is False  # high finding
    assert surf.bypassed is True

    audits, hits = finalize()
    assert audits == 3
    assert hits == 0


def test_per_entry_handler_caches_duplicate_content_hashes(monkeypatch, tmp_path):
    """Same-content mutation across path+surface shares one audit call."""
    twin = _variant("twin")
    other = _variant("other")

    call_log: list[str] = []

    def fake_call_blue_team(variant, repo, method, fn, *, sandbox_container):
        call_log.append(variant.metadata["tag"])
        return []

    monkeypatch.setattr(runner_mod._seams, "call_blue_team", fake_call_blue_team)

    on_path, on_surface, finalize = _make_handler(tmp_path)
    on_path(_path_row(twin, lethal=True, score=1.0))
    on_path(_path_row(other, lethal=True, score=1.0))
    on_surface(_surface_row(twin, lethal=True, post_hoc_ok=True))

    audits, hits = finalize()
    assert audits == 2
    assert hits == 1
    assert call_log.count("twin") == 1
    assert call_log.count("other") == 1


def test_per_entry_handler_zero_calls_yields_zero_telemetry(monkeypatch, tmp_path):
    """Never invoking the callback → zero audits, zero hits."""
    fake = MagicMock()
    monkeypatch.setattr(runner_mod._seams, "call_blue_team", fake)

    _, _, finalize = _make_handler(tmp_path)
    audits, hits = finalize()
    assert audits == 0
    assert hits == 0
    fake.assert_not_called()


def test_per_entry_handler_passes_blue_team_fn_seam(tmp_path):
    """``blue_team_fn`` overrides the production audit path (test seam)."""
    fake_fn = MagicMock(return_value=[])
    v = _variant("seam")

    on_path, _, finalize = _make_handler(tmp_path, blue_team_fn=fake_fn)
    row = _path_row(v, lethal=True, score=1.0)
    on_path(row)

    fake_fn.assert_called_once()
    audits, hits = finalize()
    assert audits == 1
    assert hits == 0
    assert row.bypassed is True


def test_per_entry_handler_writes_log_row(monkeypatch, tmp_path):
    """Each callback invocation flushes one row to log.jsonl immediately."""
    monkeypatch.setattr(
        runner_mod._seams, "call_blue_team",
        lambda *a, **kw: [],
    )
    on_path, on_surface, _ = _make_handler(tmp_path)
    on_path(_path_row(_variant("p1"), lethal=True, score=1.0))
    on_surface(_surface_row(_variant("s1"), lethal=True, post_hoc_ok=True))

    log_path = tmp_path / "log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # one per callback invocation, written immediately

    import json
    rows = [json.loads(line) for line in lines]
    phases = [r["phase"] for r in rows]
    assert phases == ["path_expansion", "surface_expansion"]
    assert all(r["is_expansion"] for r in rows)


def test_per_entry_handler_audit_failure_does_not_crash(monkeypatch, tmp_path):
    """If call_blue_team raises, entry.bypassed=None and loop continues."""
    def boom(*a, **kw):
        raise RuntimeError("audit crash")

    monkeypatch.setattr(runner_mod._seams, "call_blue_team", boom)

    on_path, _, finalize = _make_handler(tmp_path)
    row = _path_row(_variant("crash"), lethal=True, score=1.0)
    on_path(row)
    # Crash is swallowed; bypassed reset to None; counters not incremented.
    assert row.bypassed is None
    audits, hits = finalize()
    assert audits == 0
    assert hits == 0
