"""Schema DTO tests for src/red_team/log/schema.py.

The schema module defines three frozen dataclasses that represent the three
JSONL log streams the red-team writes. The tests below pin down:

* stage naming must be the semantic vocabulary (evasion / path_expansion /
  surface_expansion); the legacy S1/S2/S3 codes are explicitly rejected.
* rows are immutable after construction (frozen=True).
* ``CampaignLogRow`` permits ``round=None`` when ``is_expansion=True``
  because expansion rows are outside the per-round counter.
* ``SummaryRow`` and ``AttackLogRow`` accept their declared fields without
  validation surprises.
"""
from __future__ import annotations

import dataclasses

import pytest

from red_team.log.schema import (
    AttackLogRow,
    CampaignLogRow,
    SummaryRow,
)


# --------------------------------------------------------------------- stage

def _make_campaign_row(**overrides):
    base = dict(
        run_id="run-001",
        round=1,
        stage="evasion",
        is_expansion=False,
        timestamp="2025-01-01T00:00:00Z",
        seed_id="seed-001",
        repo_id="repo-001",
        target_model="claude-sonnet",
        bypass=False,
        lethal=False,
    )
    base.update(overrides)
    return CampaignLogRow(**base)


@pytest.mark.parametrize("stage", ["evasion", "path_expansion", "surface_expansion"])
def test_campaign_log_row_accepts_semantic_stages(stage: str) -> None:
    row = _make_campaign_row(stage=stage)
    assert row.stage == stage


@pytest.mark.parametrize("bad_stage", ["S1", "S2", "S3", "s1", "", "EVASION", "unknown"])
def test_campaign_log_row_rejects_non_semantic_stage(bad_stage: str) -> None:
    with pytest.raises(ValueError):
        _make_campaign_row(stage=bad_stage)


def test_campaign_log_row_valid_stages_exactly_three() -> None:
    # Guard against someone sneaking S1/S2/S3 back into the allowed set.
    valid = CampaignLogRow._VALID_STAGES
    assert valid == frozenset({"evasion", "path_expansion", "surface_expansion"})
    assert "S1" not in valid
    assert "S2" not in valid
    assert "S3" not in valid


# --------------------------------------------------------------------- frozen

def test_campaign_log_row_is_frozen() -> None:
    row = _make_campaign_row()
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.bypass = True  # type: ignore[misc]


def test_summary_row_is_frozen() -> None:
    row = SummaryRow(
        run_id="run-001",
        rounds_executed=5,
        first_bypass_round=3,
        first_lethal_round=None,
        total_cost_usd=1.23,
        stop_reason="budget_exhausted",
        terminated_at="2025-01-01T00:10:00Z",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.rounds_executed = 99  # type: ignore[misc]


def test_attack_log_row_is_frozen() -> None:
    row = AttackLogRow(
        run_id="run-001",
        round=1,
        seed_id="seed-001",
        repo_id="repo-001",
        variant_hash="deadbeef01234567",
        findings_summary="blocked path traversal via ../..",
        tool_chain=("bash", "read_file"),
        bypass=False,
        lethal=False,
        timestamp="2025-01-01T00:00:00Z",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.bypass = True  # type: ignore[misc]


# ------------------------------------------------------------------ expansion

def test_campaign_log_row_expansion_allows_round_none() -> None:
    row = _make_campaign_row(
        round=None,
        is_expansion=True,
        stage="path_expansion",
    )
    assert row.round is None
    assert row.is_expansion is True


def test_campaign_log_row_default_extra_is_empty_dict() -> None:
    row = _make_campaign_row()
    assert row.extra == {}


def test_campaign_log_row_optional_stop_reason_defaults_to_none() -> None:
    row = _make_campaign_row()
    assert row.stop_reason is None


# ------------------------------------------------------------------ other DTOs

def test_summary_row_allows_nullable_bypass_and_lethal_rounds() -> None:
    row = SummaryRow(
        run_id="run-001",
        rounds_executed=10,
        first_bypass_round=None,
        first_lethal_round=None,
        total_cost_usd=0.0,
        stop_reason="converged",
        terminated_at="2025-01-01T01:00:00Z",
    )
    assert row.first_bypass_round is None
    assert row.first_lethal_round is None


def test_attack_log_row_tool_chain_is_tuple() -> None:
    row = AttackLogRow(
        run_id="run-001",
        round=2,
        seed_id="seed-001",
        repo_id="repo-001",
        variant_hash="cafef00d11223344",
        findings_summary="ok",
        tool_chain=("bash",),
        bypass=True,
        lethal=False,
        timestamp="2025-01-01T00:00:00Z",
    )
    assert isinstance(row.tool_chain, tuple)
    assert row.tool_chain == ("bash",)
