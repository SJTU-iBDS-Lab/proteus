"""Tests for ``red_team.log.harvest_writer``.

Covers:
* Round-trip: build → append → re-read → equal.
* Hard rejection of ``bypass=None`` at write time.
* Builder helpers wrap SkillVariant / HoneypotResult shapes correctly.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.schemas import AttackObjective, Channel, SkillVariant, Topology

from red_team.log.harvest import HarvestRow
from red_team.log.harvest_writer import (
    append_harvest_row,
    build_expansion_harvest_row,
    build_main_round_harvest_row,
    iso_now,
)


def _make_variant(seed_id: str = "seed-04") -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.PRIVILEGE_ESCALATION,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=[Channel.FILESYSTEM],
        d="test description",
        chain=["slopbuster"],
        metadata={"seed_id": seed_id},
    )


def test_main_round_builder_round_trip(tmp_path: Path):
    parent = _make_variant()
    child = _make_variant()
    row = build_main_round_harvest_row(
        run_id="r1", seed_id="seed-04", round_index=1, stage="evasion",
        parent_variant=parent, s_prime=child,
        parent_bypass=True, parent_findings=[],
        lethal=False, target_model="gpt-5.4-mini",
        tool_chain=[{"name": "Read"}],
        sandbox_files_modified=["/tmp/x"],
        sandbox_commands_executed=["ls"],
    )
    path = tmp_path / "harvest.jsonl"
    append_harvest_row(path, row)

    line = path.read_text().strip()
    back = HarvestRow.from_jsonl_line(line)
    assert back.run_id == "r1"
    assert back.seed_id == "seed-04"
    assert back.round == 1
    assert back.bypass is True
    assert back.lethal is False
    assert back.audit_source == "skillvetter:parent_audit"
    assert back.target_model == "gpt-5.4-mini"
    assert back.tool_chain == ({"name": "Read"},)


def test_expansion_builder_round_trip(tmp_path: Path):
    parent = _make_variant()
    child = _make_variant()
    hp = SimpleNamespace(
        files_modified=["/etc/foo"],
        files_deleted=[],
        commands_executed=["chmod"],
    )
    entry = SimpleNamespace(
        mutation=child, bypassed=True, lethal=True, hp=hp,
        rationale="sibling-from-brainstorm", round=2,
    )
    row = build_expansion_harvest_row(
        run_id="r1", seed_id="seed-04",
        parent_variant=parent, entry=entry,
        stage="path_expansion", target_model="gpt-5.4-mini",
    )
    path = tmp_path / "harvest.jsonl"
    append_harvest_row(path, row)

    back = HarvestRow.from_jsonl_line(path.read_text().strip())
    assert back.stage == "path_expansion"
    assert back.bypass is True
    assert back.lethal is True
    assert back.audit_source == "skillvetter:expansion_reaudit"
    assert "/etc/foo" in back.sandbox_files_modified


def test_expansion_builder_rejects_unaudited_entry():
    parent = _make_variant()
    child = _make_variant()
    entry = SimpleNamespace(mutation=child, bypassed=None, lethal=False, hp=None)
    with pytest.raises(ValueError, match="bypassed=None"):
        build_expansion_harvest_row(
            run_id="r1", seed_id="seed-04",
            parent_variant=parent, entry=entry,
            stage="path_expansion", target_model="gpt-5.4-mini",
        )


def test_append_writer_appends(tmp_path: Path):
    parent = _make_variant()
    child = _make_variant()
    row1 = build_main_round_harvest_row(
        run_id="r1", seed_id="s", round_index=1, stage="evasion",
        parent_variant=parent, s_prime=child,
        parent_bypass=False, parent_findings=[], lethal=False,
        target_model="m", tool_chain=[],
        sandbox_files_modified=[], sandbox_commands_executed=[],
    )
    row2 = build_main_round_harvest_row(
        run_id="r1", seed_id="s", round_index=2, stage="evasion",
        parent_variant=parent, s_prime=child,
        parent_bypass=True, parent_findings=[], lethal=True,
        target_model="m", tool_chain=[],
        sandbox_files_modified=[], sandbox_commands_executed=[],
    )
    path = tmp_path / "out" / "harvest.jsonl"   # tests dir creation
    append_harvest_row(path, row1)
    append_harvest_row(path, row2)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["round"] == 1
    assert json.loads(lines[1])["round"] == 2


def test_iso_now_format():
    s = iso_now()
    assert s.endswith("Z")
    assert "T" in s
    # YYYY-MM-DDTHH:MM:SSZ → 20 chars
    assert len(s) == 20
