"""Unit tests for :mod:`red_team.campaign.io` — log row assembly + writers.

These tests pin the dict-schema produced by ``build_log_row`` and the
wrapper ``write_log_row`` / ``append_attack_log`` / ``write_summary``
helpers used by the campaign runner.  They are deliberately small so a
schema regression surfaces here before the integration suite runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
from red_team.campaign import io as campaign_io
from sandbox.honeypot import HoneypotResult


def _variant() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=[Channel.FILESYSTEM],
        d="doc",
        chain=["skill-alpha"],
        metadata={"seed_id": "seed-001"},
    )


def _finding(sev: str = "low") -> PaperFinding:
    return PaperFinding(
        finding_id=f"x-{sev}",
        layer="A",
        category="guidance_injection",
        reason_code="autonomy_encouragement",
        confidence=0.8,
        severity=sev,
        evidence=Evidence(snippet=""),
        explanation="test",
    )


def test_build_log_row_has_required_keys():
    """Every field the smoke suite asserts on must be present."""
    row = campaign_io.build_log_row(
        round_index=1,
        variant=_variant(),
        findings=[_finding("low")],
        hp=None,
        bypassed=True,
        lethal_flag=False,
        backend="agent",
        stop_reason="submitted",
        rationale="ok",
        tool_chain=[{"tool": "read_skill"}],
        cost_usd=0.01,
        is_expansion=False,
        phase="evasion",
    )
    expected = {
        "round", "variant_hash", "o", "tau", "chain", "bypassed", "lethal",
        "backend", "stop_reason", "rationale", "tool_chain", "findings_count",
        "findings", "cost_usd", "is_expansion", "phase", "forced_topology",
        "chain_original", "chain_used", "hp", "timestamp",
    }
    assert expected.issubset(row.keys())
    assert row["round"] == 1
    assert row["backend"] == "agent"
    assert row["phase"] == "evasion"
    assert row["bypassed"] is True
    assert row["forced_topology"] is False
    assert row["findings_count"] == 1


def test_build_log_row_variant_none_produces_nullish():
    row = campaign_io.build_log_row(
        round_index=None,
        variant=None,
        findings=[],
        hp=None,
        bypassed=False,
        lethal_flag=False,
        backend="agent",
        stop_reason="refusal",
        rationale="",
        tool_chain=[],
        cost_usd=0.0,
        is_expansion=False,
        phase="evasion",
    )
    assert row["variant_hash"] is None
    assert row["chain"] is None
    assert row["o"] is None
    assert row["tau"] is None


def test_write_log_row_appends_jsonl(tmp_path):
    log_path = tmp_path / "log.jsonl"
    r1 = campaign_io.build_log_row(
        round_index=1, variant=_variant(), findings=[], hp=None,
        bypassed=False, lethal_flag=False, backend="agent",
        stop_reason="submitted", rationale="", tool_chain=[],
        cost_usd=0.0, is_expansion=False, phase="evasion",
    )
    campaign_io.write_log_row(log_path, r1)
    r2 = dict(r1, round=2)
    campaign_io.write_log_row(log_path, r2)
    lines = [
        json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(lines) == 2
    assert lines[0]["round"] == 1 and lines[1]["round"] == 2


def test_build_summary_has_expected_keys():
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[4] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from red_team.campaign.run_config import RunConfig
    rc = RunConfig()
    summary = campaign_io.build_summary(
        run_id="rid",
        seed_id="seed-001",
        repo_id="r1",
        target_model="m",
        mutator_model="m2",
        rounds_ran=3,
        first_bypass_round=2,
        total_bypass=1,
        total_lethal=1,
        total_cost=0.123456789,
        run_config=rc,
    )
    expected = {
        "run_id", "seed_id", "repo_id", "target_model", "mutator_model",
        "rounds_ran", "first_bypass_round", "total_bypass", "total_lethal",
        "total_cost_usd", "ablation",
    }
    assert expected.issubset(summary.keys())
    assert summary["total_cost_usd"] == round(0.123456789, 6)
    assert isinstance(summary["ablation"], dict)


def test_coerce_tool_chain_normalizes_to_list():
    assert campaign_io._coerce_tool_chain(None) == []
    assert campaign_io._coerce_tool_chain([1, 2]) == [1, 2]
    assert campaign_io._coerce_tool_chain({"x": 1}) == [{"x": 1}]


def test_build_log_row_preserves_none_bypassed():
    """``bypassed=None`` survives row build (expansion-row tri-state)."""
    row = campaign_io.build_log_row(
        round_index=None,
        variant=_variant(),
        findings=[],
        hp=None,
        bypassed=None,
        lethal_flag=True,
        backend="agent",
        stop_reason="",
        rationale="",
        tool_chain=[],
        cost_usd=0.0,
        is_expansion=True,
        phase="path_expansion",
    )
    assert row["bypassed"] is None
    assert row["lethal"] is True


def test_build_summary_includes_expansion_audit_telemetry():
    """The post-A1 summary surfaces audits/seed and cache-hit counts."""
    import sys
    from pathlib import Path
    src = Path(__file__).resolve().parents[4] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from red_team.campaign.run_config import RunConfig
    summary = campaign_io.build_summary(
        run_id="rid", seed_id="seed-001", repo_id="r1",
        target_model="m", mutator_model="m2", rounds_ran=3,
        first_bypass_round=2, total_bypass=1, total_lethal=1,
        total_cost=0.0, run_config=RunConfig(),
        expansion_audits=8, expansion_audit_cache_hits=2,
    )
    assert summary["expansion_audits"] == 8
    assert summary["expansion_audit_cache_hits"] == 2
