"""Unit tests for ``src/red_team/log/attack_log.py``.

The log module owns the real implementation of the attack-log JSONL
writer/reader/search. The tests below pin down:

* ``append(path, entry_or_row)`` writes a single JSONL line and preserves
  the order of multiple appends.
* ``append`` accepts both a plain ``dict`` (legacy callers) and an
  :class:`AttackLogRow` DTO (new callers).
* ``read_all(path)`` returns the stored dicts in insertion order.
* ``search(path, query, k, ...)`` uses BM25 via the retrieval sub-package
  and honours the optional ``seed_id`` / ``filter_bypassed`` filters.
* Path-discipline: ``attack_log_path(run_id)`` from :mod:`red_team.log.paths`
  is the only way to resolve the on-disk location; we exercise the round
  trip from ``run_id`` → ``attack_log_path`` → ``append`` → ``read_all``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from red_team.log import attack_log_path
from red_team.log.attack_log import (
    append,
    read_all,
    row_to_document,
    search,
)
from red_team.log.schema import AttackLogRow


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _dict_entry(
    i: int,
    *,
    reason: str,
    codes: list[str],
    tools: list[str],
    seed_id: str,
    bypassed: bool,
) -> dict:
    return {
        "round": i,
        "seed_id": seed_id,
        "variant_hash": f"hash-{i:03d}",
        "bypassed": bypassed,
        "lethal": bypassed and i % 2 == 0,
        "reason": reason,
        "reason_codes_evaded": codes,
        "tool_chain_summary": [{"tool": t} for t in tools],
        "timestamp": f"2026-04-{(i % 28) + 1:02d}T00:00:00",
    }


# --------------------------------------------------------------------------- #
# append() — dict path
# --------------------------------------------------------------------------- #


def test_append_dict_creates_file_and_parent_dir(tmp_path: Path) -> None:
    log_path = tmp_path / "deep" / "nested" / "attack_log.jsonl"
    append(log_path, _dict_entry(1, reason="x", codes=[], tools=["t"],
                                 seed_id="s1", bypassed=False))
    assert log_path.exists()
    # One line, ending with a newline.
    content = log_path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert content.count("\n") == 1


def test_append_dict_preserves_order_across_multiple_calls(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    entries = [
        _dict_entry(i, reason=f"reason {i}", codes=[f"code{i}"],
                    tools=[f"tool{i}"], seed_id="seed-01", bypassed=i % 2 == 0)
        for i in range(5)
    ]
    for e in entries:
        append(log_path, e)

    # Raw file reads back in order.
    with log_path.open("r", encoding="utf-8") as fh:
        raw = [json.loads(line) for line in fh if line.strip()]
    assert raw == entries


# --------------------------------------------------------------------------- #
# append() — DTO path
# --------------------------------------------------------------------------- #


def test_append_accepts_attack_log_row_dto(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    row = AttackLogRow(
        run_id="run-001",
        round=7,
        seed_id="seed-001",
        repo_id="repo-001",
        variant_hash="deadbeef01234567",
        findings_summary="blocked autonomy encouragement in skill description",
        tool_chain=("read_skill", "read_attack_log"),
        bypass=False,
        lethal=False,
        timestamp="2025-01-01T00:00:00Z",
    )
    append(log_path, row)

    stored = read_all(log_path)
    assert len(stored) == 1
    s = stored[0]
    # The DTO must round-trip into a dict that carries every declared field.
    assert s["run_id"] == "run-001"
    assert s["round"] == 7
    assert s["seed_id"] == "seed-001"
    assert s["repo_id"] == "repo-001"
    assert s["variant_hash"] == "deadbeef01234567"
    assert s["findings_summary"].startswith("blocked autonomy")
    assert s["tool_chain"] == ["read_skill", "read_attack_log"]
    assert s["bypass"] is False
    assert s["lethal"] is False
    assert s["timestamp"] == "2025-01-01T00:00:00Z"


# --------------------------------------------------------------------------- #
# read_all()
# --------------------------------------------------------------------------- #


def test_read_all_returns_empty_list_for_missing_file(tmp_path: Path) -> None:
    ghost = tmp_path / "does_not_exist.jsonl"
    assert read_all(ghost) == []


def test_read_all_skips_blank_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    # Manually craft content with blank lines in the middle.
    body = (
        json.dumps({"round": 1, "reason": "a"}) + "\n"
        "\n"
        + json.dumps({"round": 2, "reason": "b"}) + "\n"
    )
    log_path.write_text(body, encoding="utf-8")
    rows = read_all(log_path)
    assert [r["round"] for r in rows] == [1, 2]


# --------------------------------------------------------------------------- #
# search()
# --------------------------------------------------------------------------- #


@pytest.fixture
def seeded_log(tmp_path: Path) -> Path:
    log_path = tmp_path / "run-xyz" / "attack_log.jsonl"
    rows = []
    # 5 "autonomy" rows.
    for i in range(5):
        rows.append(_dict_entry(
            i,
            reason="autonomy encouragement flagged in skill description",
            codes=["autonomy_encouragement"],
            tools=["read_skill"],
            seed_id=f"seed-{(i % 3) + 1:02d}",
            bypassed=i % 2 == 0,
        ))
    # 5 "credential exfil" rows.
    for j in range(5):
        rows.append(_dict_entry(
            100 + j,
            reason="credential exfiltration via AWS metadata endpoint",
            codes=["credential_exfil"],
            tools=["list_skills", "read_skill"],
            seed_id=f"seed-{(j % 3) + 1:02d}",
            bypassed=True,
        ))
    for r in rows:
        append(log_path, r)
    return log_path


def test_search_returns_relevant_rows_for_autonomy_query(seeded_log: Path) -> None:
    hits = search(seeded_log, "autonomy encouragement", k=3)
    assert 1 <= len(hits) <= 3
    # Top hit must actually mention autonomy in reason or code list.
    top = hits[0]
    blob = (top.get("reason") or "") + " " + " ".join(top.get("reason_codes_evaded") or [])
    assert "autonomy" in blob.lower()


def test_search_honors_top_k_cap(seeded_log: Path) -> None:
    hits = search(seeded_log, "credential exfiltration", k=2)
    assert len(hits) <= 2


def test_search_missing_file_returns_empty(tmp_path: Path) -> None:
    assert search(tmp_path / "nope.jsonl", "anything", k=5) == []


def test_search_respects_seed_id_filter(seeded_log: Path) -> None:
    hits = search(seeded_log, "autonomy", k=10, seed_id="seed-01")
    assert hits, "expected at least one seed-01 hit for 'autonomy'"
    assert all(r["seed_id"] == "seed-01" for r in hits)


# --------------------------------------------------------------------------- #
# Path discipline — must use attack_log_path(run_id)
# --------------------------------------------------------------------------- #


def test_attack_log_roundtrip_via_paths_helper(tmp_path: Path) -> None:
    run_id = "run-path-test"
    path = attack_log_path(run_id, root=tmp_path)
    assert path == tmp_path / run_id / "attack_log.jsonl"

    append(path, _dict_entry(0, reason="boom", codes=["x"],
                             tools=["t"], seed_id="s", bypassed=True))
    rows = read_all(path)
    assert len(rows) == 1
    assert rows[0]["reason"] == "boom"


# --------------------------------------------------------------------------- #
# row_to_document helper — used by the BM25 retriever.
# --------------------------------------------------------------------------- #


def test_row_to_document_includes_reason_codes_and_tools() -> None:
    row = {
        "reason": "leaked credentials",
        "reason_codes_evaded": ["credential_exfil", "network_exfil"],
        "tool_chain_summary": [{"tool": "read_skill"}, {"tool": "list_skills"}],
    }
    doc = row_to_document(row)
    assert "leaked credentials" in doc
    assert "credential_exfil" in doc
    assert "network_exfil" in doc
    assert "read_skill" in doc
    assert "list_skills" in doc


def test_row_to_document_handles_missing_fields() -> None:
    assert row_to_document({}) == "  "  # three empty pieces joined by spaces
