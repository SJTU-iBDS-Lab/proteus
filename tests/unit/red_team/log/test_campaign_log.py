"""Unit tests for :mod:`red_team.log.campaign_log` — W3 extraction.

The campaign log writer replaces driver.py's inline ``json.dumps`` +
``path.open("a")`` block. It must be byte-equivalent for dict-as-row
callers (the sole legacy channel) and accept the ``CampaignLogRow`` DTO
via ``asdict``.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from red_team.log import campaign_log
from red_team.log.schema import CampaignLogRow


# ---------------------------------------------------------------------------
# now_iso()
# ---------------------------------------------------------------------------


def test_now_iso_returns_iso_seconds_string():
    ts = campaign_log.now_iso()
    assert isinstance(ts, str)
    # e.g. '2026-04-23T11:22:33+00:00'
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\+00:00|Z)$", ts
    ), f"bad iso-8601 seconds-resolution timestamp: {ts!r}"


# ---------------------------------------------------------------------------
# write_row + dict-as-row
# ---------------------------------------------------------------------------


def test_write_row_dict_creates_file_with_json_and_newline(tmp_path):
    path = tmp_path / "log.jsonl"
    row = {"round": 1, "bypassed": False, "timestamp": "2026-01-01T00:00:00+00:00"}

    campaign_log.write_row(path, row)

    assert path.exists()
    content = path.read_text(encoding="utf-8")
    # exactly one line with terminating newline
    assert content.endswith("\n")
    assert content.count("\n") == 1
    parsed = json.loads(content)
    assert parsed == row


def test_write_row_dict_appends_without_overwriting(tmp_path):
    path = tmp_path / "log.jsonl"
    row1 = {"round": 1}
    row2 = {"round": 2}

    campaign_log.write_row(path, row1)
    campaign_log.write_row(path, row2)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == row1
    assert json.loads(lines[1]) == row2


def test_write_row_creates_missing_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "log.jsonl"
    assert not path.parent.exists()

    campaign_log.write_row(path, {"round": 1})

    assert path.exists()


# ---------------------------------------------------------------------------
# write_row + DTO-as-row
# ---------------------------------------------------------------------------


def test_write_row_dto_serializes_via_asdict(tmp_path):
    path = tmp_path / "log.jsonl"
    dto = CampaignLogRow(
        run_id="run-001",
        round=1,
        stage="evasion",
        is_expansion=False,
        timestamp="2026-01-01T00:00:00+00:00",
        seed_id="seed-a",
        repo_id="repo-a",
        target_model="claude-sonnet-4-5",
        bypass=False,
        lethal=False,
    )

    campaign_log.write_row(path, dto)

    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["run_id"] == "run-001"
    assert parsed["stage"] == "evasion"
    assert parsed["bypass"] is False
    assert parsed["lethal"] is False
    # extra should also be present (even if empty dict)
    assert parsed["extra"] == {}


# ---------------------------------------------------------------------------
# read_rows
# ---------------------------------------------------------------------------


def test_read_rows_missing_path_returns_empty_list(tmp_path):
    missing = tmp_path / "does-not-exist.jsonl"
    assert campaign_log.read_rows(missing) == []


def test_read_rows_deserializes_multiple_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    rows_in = [
        {"round": 1, "bypassed": True},
        {"round": 2, "bypassed": False},
        {"round": 3, "bypassed": True},
    ]
    for r in rows_in:
        campaign_log.write_row(path, r)

    rows_out = campaign_log.read_rows(path)

    assert rows_out == rows_in


def test_write_then_read_round_trip_preserves_dict(tmp_path):
    path = tmp_path / "log.jsonl"
    row = {
        "round": 7,
        "o": "persistence",
        "tau": "pair",
        "chain": ["skill-alpha", "skill-beta"],
        "bypassed": True,
        "lethal": False,
        "findings_count": 3,
        "tool_chain": [{"tool": "read_skill", "args": {}}],
    }

    campaign_log.write_row(path, row)
    rows = campaign_log.read_rows(path)

    assert rows == [row]


def test_read_rows_skips_blank_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(
        '{"round": 1}\n\n{"round": 2}\n   \n',
        encoding="utf-8",
    )

    rows = campaign_log.read_rows(path)

    assert rows == [{"round": 1}, {"round": 2}]
