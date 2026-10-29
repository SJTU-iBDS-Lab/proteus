"""Unit tests for :mod:`red_team.log.summary` — W3 extraction.

The summary writer replaces driver.py's inline
``path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))``.
It must preserve the indent=2 pretty format so paper-analysis scripts
that diff summary.json across runs continue to line up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from red_team.log import summary as summary_mod
from red_team.log.schema import SummaryRow


def test_write_summary_uses_indent_and_round_trips(tmp_path):
    path = tmp_path / "summary.json"
    payload = {
        "run_id": "r-001",
        "rounds_ran": 3,
        "first_bypass_round": 2,
        "total_cost_usd": 0.42,
    }

    summary_mod.write_summary(path, payload)

    raw = path.read_text(encoding="utf-8")
    # indent=2 means multi-line output; single-line would be a regression
    assert "\n" in raw
    assert '  "run_id"' in raw, "expected 2-space indent in summary.json"
    # round-trip semantically
    assert json.loads(raw) == payload


def test_write_summary_accepts_summary_row_dto(tmp_path):
    path = tmp_path / "summary.json"
    dto = SummaryRow(
        run_id="r-002",
        rounds_executed=5,
        first_bypass_round=3,
        first_lethal_round=4,
        total_cost_usd=1.23,
        stop_reason="bypass_and_lethal",
        terminated_at="2026-01-01T00:00:00+00:00",
    )

    summary_mod.write_summary(path, dto)

    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["run_id"] == "r-002"
    assert parsed["rounds_executed"] == 5
    assert parsed["first_bypass_round"] == 3
    assert parsed["stop_reason"] == "bypass_and_lethal"


def test_write_summary_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "summary.json"
    assert not path.parent.exists()

    summary_mod.write_summary(path, {"run_id": "x"})

    assert path.exists()


def test_read_summary_round_trips(tmp_path):
    path = tmp_path / "summary.json"
    payload = {"run_id": "r-004", "total_bypass": 2, "nested": {"k": [1, 2, 3]}}

    summary_mod.write_summary(path, payload)
    got = summary_mod.read_summary(path)

    assert got == payload
