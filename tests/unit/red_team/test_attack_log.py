"""Unit tests for `src/red_team/agent/attack_log.py`.

Covers the frozen API contract:

- ``append(path, entry)``: append-only JSONL, creates file, preserves order.
- ``search(path, query, k, filter_bypassed=None, seed_id=None)``:
  BM25 retrieval with optional pre-filters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from red_team.agent.attack_log import append, search


# --------------------------------------------------------------------------- #
# Fixtures: 50 synthetic attack-log rows with varied content.
# --------------------------------------------------------------------------- #


def _make_entry(
    i: int,
    *,
    reason: str,
    reason_codes_evaded: list[str],
    tools: list[str],
    seed_id: str,
    bypassed: bool,
) -> dict:
    return {
        "round": i,
        "seed_id": seed_id,
        "variant_hash": f"hash-{i:03d}",
        "o": "exfiltrate_aws_credentials" if i % 2 == 0 else "privilege_escalation",
        "tau": "linear" if i % 3 else "dag",
        "bypassed": bypassed,
        "lethal": bypassed and (i % 4 != 0),
        "reason": reason,
        "reason_codes_evaded": reason_codes_evaded,
        "tool_chain_summary": [{"tool": t} for t in tools],
        "timestamp": f"2026-04-{(i % 28) + 1:02d}T12:00:00",
    }


def _build_synthetic_entries() -> list[dict]:
    entries: list[dict] = []

    # 15 rows heavy on "autonomy" semantics.
    for i in range(15):
        entries.append(
            _make_entry(
                i,
                reason=(
                    "Scanner flagged autonomy encouragement in skill "
                    "description; rewritten to keep autonomy implicit."
                ),
                reason_codes_evaded=["autonomy_encouragement", "scope_creep"],
                tools=["read_skill", "read_attack_log"],
                seed_id=f"seed-{(i % 5) + 1:02d}",
                bypassed=(i % 2 == 0),
            )
        )

    # 15 rows heavy on shell/rm-rf semantics.
    for j in range(15):
        i = 100 + j
        entries.append(
            _make_entry(
                i,
                reason=(
                    "Destructive shell invocation rm -rf /tmp/cache flagged "
                    "as dangerous command"
                ),
                reason_codes_evaded=["dangerous_command", "shell_exec"],
                tools=["read_repo_context", "submit_mutation"],
                seed_id=f"seed-{(j % 5) + 1:02d}",
                bypassed=(j % 3 == 0),
            )
        )

    # 15 rows about credential exfiltration.
    for j in range(15):
        i = 200 + j
        entries.append(
            _make_entry(
                i,
                reason=(
                    "Credential exfiltration attempt via AWS metadata "
                    "endpoint masked as telemetry"
                ),
                reason_codes_evaded=["credential_exfil", "network_exfil"],
                tools=["read_skill", "list_skills"],
                seed_id=f"seed-{(j % 5) + 1:02d}",
                bypassed=(j % 2 == 1),
            )
        )

    # 5 rows tagged seed-04 explicitly so seed filter has enough rows.
    for j in range(5):
        i = 300 + j
        entries.append(
            _make_entry(
                i,
                reason="Mixed-signal round with benign language and latent exec",
                reason_codes_evaded=["latent_exec"],
                tools=["read_attack_log"],
                seed_id="seed-04",
                bypassed=True,
            )
        )

    assert len(entries) == 50
    return entries


@pytest.fixture
def populated_log(tmp_path: Path) -> Path:
    log_path = tmp_path / "nested" / "attack_log.jsonl"
    for entry in _build_synthetic_entries():
        append(log_path, entry)
    assert log_path.exists()
    return log_path


# --------------------------------------------------------------------------- #
# append() tests
# --------------------------------------------------------------------------- #


def test_append_creates_file_and_preserves_order(tmp_path: Path) -> None:
    log_path = tmp_path / "sub" / "log.jsonl"
    entries = _build_synthetic_entries()

    for entry in entries:
        append(log_path, entry)

    assert log_path.exists()
    # Read back by parsing each JSONL line — same order as written.
    with log_path.open("r", encoding="utf-8") as fh:
        read_back = [json.loads(line) for line in fh if line.strip()]

    assert read_back == entries


def test_append_writes_trailing_newline_per_line(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    append(log_path, {"round": 1, "reason": "a"})
    append(log_path, {"round": 2, "reason": "b"})

    content = log_path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert content.count("\n") == 2


# --------------------------------------------------------------------------- #
# search() tests
# --------------------------------------------------------------------------- #


def test_search_autonomy_returns_relevant_top_results(populated_log: Path) -> None:
    results = search(populated_log, "autonomy", k=5)
    assert 1 <= len(results) <= 5

    # Top-1 must contain "autonomy" in reason or reason_codes_evaded.
    top = results[0]
    combined = (
        (top.get("reason") or "").lower()
        + " "
        + " ".join(top.get("reason_codes_evaded") or []).lower()
    )
    assert "autonomy" in combined

    # All returned rows should mention autonomy in reason or codes.
    for row in results:
        blob = (
            (row.get("reason") or "").lower()
            + " "
            + " ".join(row.get("reason_codes_evaded") or []).lower()
        )
        assert "autonomy" in blob or "scope_creep" in blob


def test_search_rm_rf_returns_distinct_set(populated_log: Path) -> None:
    autonomy_hits = search(populated_log, "autonomy", k=5)
    rm_hits = search(populated_log, "rm -rf", k=5)

    assert rm_hits, "expected at least one hit for 'rm -rf'"
    top = rm_hits[0]
    top_reason = (top.get("reason") or "").lower()
    assert "rm" in top_reason or "rf" in top_reason

    # The two result sets must not be identical.
    autonomy_hashes = {r["variant_hash"] for r in autonomy_hits}
    rm_hashes = {r["variant_hash"] for r in rm_hits}
    assert autonomy_hashes != rm_hashes


def test_search_filter_bypassed_true_only_keeps_bypassed_rows(
    populated_log: Path,
) -> None:
    results = search(
        populated_log, "credential exfiltration", k=10, filter_bypassed=True
    )
    assert results, "filter_bypassed=True should still return hits for credential query"
    assert all(row["bypassed"] is True for row in results)


def test_search_seed_id_filter_restricts_to_seed(populated_log: Path) -> None:
    results = search(populated_log, "autonomy exfiltration rm", k=10, seed_id="seed-04")
    assert results, "seed-04 has >= 3 rows so results must be non-empty"
    assert all(row["seed_id"] == "seed-04" for row in results)


def test_search_returns_empty_when_filtered_rows_below_three(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "tiny.jsonl"
    # Only 2 rows with bypassed=True.
    append(
        log_path,
        _make_entry(
            1,
            reason="autonomy issue",
            reason_codes_evaded=["autonomy_encouragement"],
            tools=["read_skill"],
            seed_id="seed-01",
            bypassed=True,
        ),
    )
    append(
        log_path,
        _make_entry(
            2,
            reason="autonomy issue 2",
            reason_codes_evaded=["autonomy_encouragement"],
            tools=["read_skill"],
            seed_id="seed-01",
            bypassed=True,
        ),
    )
    # A non-matching row.
    append(
        log_path,
        _make_entry(
            3,
            reason="unrelated",
            reason_codes_evaded=[],
            tools=["read_skill"],
            seed_id="seed-99",
            bypassed=False,
        ),
    )

    # filter_bypassed=True → only 2 rows qualify → fewer than 3 → [].
    assert search(log_path, "autonomy", k=5, filter_bypassed=True) == []


def test_search_missing_file_returns_empty(tmp_path: Path) -> None:
    ghost = tmp_path / "does_not_exist.jsonl"
    assert search(ghost, "anything", k=5) == []
