"""Unit tests for :mod:`red_team.agent.tools` (IMPL-01).

Coverage goals:
  * ``TOOLS`` exposes exactly the 5 schemas the paper §4.2 frozen contract
    requires, with the correct names and no ``web_search``.
  * ``_dispatch_tool`` routes each known tool to a handler that reads from
    a fixture data-root (set via ``CLAW_DATA_ROOT`` env var).
  * Error paths: invalid ``section``, unknown skill_id, unknown repo_id,
    unknown tool name all behave as documented — the first three return
    ``{"error": ...}``, the last raises ``KeyError``.
  * Terminal sentinel: ``submit_mutation`` returns
    ``{"_terminal": True, "args": ...}`` without attempting to build a
    SkillVariant (that's IMPL-04's job).

The fixture data-root lives at ``tests/fixtures/mini_data/`` with the
same layout as production ``data/``, so we can point the tools at it
with a single env-var flip.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from red_team.agent.tools import (
    TOOL_NAMES,
    TOOLS,
    _dispatch_tool,
)


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "fixtures" / "mini_data"
)


@pytest.fixture()
def data_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the tools module at the fixture data-root for this test.

    The env var is read per-call inside :func:`tools._data_root`, so using
    :meth:`monkeypatch.setenv` fully isolates each test from the repo's
    real ``data/`` directory.
    """
    assert FIXTURE_ROOT.is_dir(), f"fixture root missing: {FIXTURE_ROOT}"
    monkeypatch.setenv("CLAW_DATA_ROOT", str(FIXTURE_ROOT))
    return FIXTURE_ROOT


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_tools_list_has_exactly_six_entries() -> None:
    """The red-team agent ships 6 tools: the 5 paper-§4.2 retrieval /
    terminal primitives plus ``get_binary`` for code-layer mutation.
    Schema drift would silently expand the agent's action space, so we
    guard the count hard."""
    assert len(TOOLS) == 6


def test_tool_names_match_frozen_contract_in_order() -> None:
    """Name *and* ordering are part of the contract because downstream
    prompt templates (IMPL-02) enumerate tools positionally for humans."""
    names = [t["function"]["name"] for t in TOOLS]
    assert names == [
        "read_skill",
        "list_skills",
        "read_repo_context",
        "read_attack_log",
        "get_binary",
        "submit_mutation",
    ]
    # Also asserts the module-level constant stays in sync with the schema.
    assert tuple(names) == TOOL_NAMES


def test_tools_are_openai_function_calling_shape() -> None:
    """Each entry must be ``{type: function, function: {name, description,
    parameters}}`` so an OpenAI-compatible client accepts it directly."""
    for tool in TOOLS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


def test_no_web_search_tool_is_exposed() -> None:
    """Disallowing public-web evidence is a paper claim; the schema must
    match the claim so an adversarial runtime can't reintroduce it."""
    names = {t["function"]["name"] for t in TOOLS}
    assert "web_search" not in names
    assert "browse" not in names


def test_submit_mutation_schema_requires_new_d_and_rationale() -> None:
    """The terminal tool must surface at least ``new_d`` + ``rationale``;
    the other fields (new_c, new_kappa) are optional per search-space."""
    submit = next(
        t for t in TOOLS if t["function"]["name"] == "submit_mutation"
    )
    schema = submit["function"]["parameters"]
    assert set(schema["required"]) == {"new_d", "rationale"}
    # new_c / new_kappa are declared but optional.
    assert "new_c" in schema["properties"]
    assert "new_kappa" in schema["properties"]


def test_all_tool_results_are_json_serializable(data_root: Path) -> None:
    """Every tool response feeds the LLM as a tool-message ``content``,
    which must be ``json.dumps``-able. We spot-check each handler."""
    # read_skill
    r = _dispatch_tool(
        "read_skill", {"skill_id": "demo-echo"}, None, None, None, None
    )
    json.dumps(r)
    # list_skills
    r = _dispatch_tool("list_skills", {"category": "all"}, None, None, None, None)
    json.dumps(r)
    # read_repo_context
    r = _dispatch_tool(
        "read_repo_context",
        {"repo_id": "demo-repo", "section": "readme"},
        None,
        None,
        None,
        None,
    )
    json.dumps(r)
    # read_attack_log (no path → empty hits)
    r = _dispatch_tool(
        "read_attack_log", {"query": "anything"}, None, None, None, None
    )
    json.dumps(r)
    # submit_mutation
    r = _dispatch_tool(
        "submit_mutation",
        {"new_d": "x", "rationale": "y"},
        None,
        None,
        None,
        None,
    )
    json.dumps(r)


# ---------------------------------------------------------------------------
# read_skill
# ---------------------------------------------------------------------------


def test_read_skill_benign_returns_skill_md_and_entry_script(
    data_root: Path,
) -> None:
    """Benign split reads SKILL.md + first entry script (4KB cap)."""
    result = _dispatch_tool(
        "read_skill",
        {"skill_id": "demo-echo", "split": "benign"},
        None,
        None,
        None,
        None,
    )
    assert result["skill_id"] == "demo-echo"
    assert result["split"] == "benign"
    # SKILL.md body includes the title header we wrote.
    assert "# demo-echo" in result["skill_md"]
    assert result["entry_script_name"] == "entry.py"
    assert "def main" in result["entry_script"]


def test_read_skill_defaults_to_benign(data_root: Path) -> None:
    """Omitting ``split`` should behave identically to split='benign'."""
    without = _dispatch_tool(
        "read_skill", {"skill_id": "demo-echo"}, None, None, None, None
    )
    assert without["split"] == "benign"
    assert "# demo-echo" in without["skill_md"]


def test_read_skill_benign_missing_returns_error(data_root: Path) -> None:
    """Unknown skill → structured error (LLM can self-correct) — NOT an
    exception, because we want the agent loop to continue."""
    result = _dispatch_tool(
        "read_skill",
        {"skill_id": "does-not-exist", "split": "benign"},
        None,
        None,
        None,
        None,
    )
    assert "error" in result


def test_read_skill_malicious_returns_poisoned_content(data_root: Path) -> None:
    """Malicious split finds <poisoned>/<chain_hash>/<skill_id>/SKILL.md
    and returns the chain_hash alongside the body."""
    result = _dispatch_tool(
        "read_skill",
        {"skill_id": "demo-echo", "split": "malicious"},
        None,
        None,
        None,
        None,
    )
    assert result["split"] == "malicious"
    assert result["chain_hash"] == "abc123hash"
    assert "poisoned" in result["skill_md"]


def test_read_skill_malicious_absent_returns_no_version_error(
    data_root: Path,
) -> None:
    """When no poisoned version exists anywhere, return the documented
    sentinel error string (runtime relies on the exact text)."""
    result = _dispatch_tool(
        "read_skill",
        {"skill_id": "never-poisoned", "split": "malicious"},
        None,
        None,
        None,
        None,
    )
    assert result == {"error": "no malicious version"}


def test_read_skill_invalid_split_returns_error(data_root: Path) -> None:
    result = _dispatch_tool(
        "read_skill",
        {"skill_id": "demo-echo", "split": "nope"},
        None,
        None,
        None,
        None,
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------


def test_list_skills_benign_returns_demo_echo(data_root: Path) -> None:
    result = _dispatch_tool(
        "list_skills", {"category": "benign"}, None, None, None, None
    )
    ids = [it["id"] for it in result["items"]]
    assert "demo-echo" in ids
    # One-liner was extracted from front-matter.
    demo = next(it for it in result["items"] if it["id"] == "demo-echo")
    assert "Echo the input" in demo["one_liner"]


def test_list_skills_malicious_reads_seeds_jsonl(data_root: Path) -> None:
    result = _dispatch_tool(
        "list_skills", {"category": "malicious"}, None, None, None, None
    )
    ids = [it["id"] for it in result["items"]]
    assert "seed-demo-exfil" in ids
    seed = next(it for it in result["items"] if it["id"] == "seed-demo-exfil")
    assert seed["objective"] == "data_exfiltration"
    assert seed["chain"] == ["demo-echo"]


def test_list_skills_all_concatenates_both(data_root: Path) -> None:
    result = _dispatch_tool(
        "list_skills", {"category": "all"}, None, None, None, None
    )
    splits = {it["split"] for it in result["items"]}
    assert splits == {"benign", "malicious"}


def test_list_skills_invalid_category_returns_error(data_root: Path) -> None:
    result = _dispatch_tool(
        "list_skills", {"category": "weird"}, None, None, None, None
    )
    assert "error" in result


# ---------------------------------------------------------------------------
# read_repo_context
# ---------------------------------------------------------------------------


def test_read_repo_context_readme(data_root: Path) -> None:
    result = _dispatch_tool(
        "read_repo_context",
        {"repo_id": "demo-repo", "section": "readme"},
        None,
        None,
        None,
        None,
    )
    assert result["repo_id"] == "demo-repo"
    assert result["section"] == "readme"
    assert "demo-repo" in result["content"]


def test_read_repo_context_tree_returns_list(data_root: Path) -> None:
    result = _dispatch_tool(
        "read_repo_context",
        {"repo_id": "demo-repo", "section": "tree"},
        None,
        None,
        None,
        None,
    )
    assert isinstance(result["content"], list)
    assert "README.md" in result["content"]


def test_read_repo_context_theme(data_root: Path) -> None:
    result = _dispatch_tool(
        "read_repo_context",
        {"repo_id": "demo-repo", "section": "theme"},
        None,
        None,
        None,
        None,
    )
    assert "Fixture repository" in result["content"]


def test_read_repo_context_invalid_section_returns_error(
    data_root: Path,
) -> None:
    """Contract: invalid ``section`` → ``{"error": ...}``, never a crash."""
    result = _dispatch_tool(
        "read_repo_context",
        {"repo_id": "demo-repo", "section": "nonexistent"},
        None,
        None,
        None,
        None,
    )
    assert "error" in result


def test_read_repo_context_unknown_repo_returns_error(data_root: Path) -> None:
    result = _dispatch_tool(
        "read_repo_context",
        {"repo_id": "no-such-repo", "section": "readme"},
        None,
        None,
        None,
        None,
    )
    assert "error" in result


def test_read_repo_context_falls_back_to_downloaded_skill(data_root: Path) -> None:
    result = _dispatch_tool(
        "read_repo_context",
        {"repo_id": "demo-echo", "section": "readme"},
        None,
        None,
        None,
        None,
    )
    assert result["repo_id"] == "demo-echo"
    assert result["section"] == "readme"
    assert "demo-echo" in result["content"]


# ---------------------------------------------------------------------------
# read_attack_log
# ---------------------------------------------------------------------------


def test_read_attack_log_no_path_returns_empty_hits(data_root: Path) -> None:
    """Round 0 / memory_enabled=False case — no log path supplied."""
    result = _dispatch_tool(
        "read_attack_log", {"query": "autonomy"}, None, None, None, None
    )
    assert result["hits"] == []
    assert result["query"] == "autonomy"


def test_read_attack_log_missing_query_returns_error(data_root: Path) -> None:
    result = _dispatch_tool(
        "read_attack_log", {}, None, None, None, None
    )
    assert "error" in result


def test_read_attack_log_impl03_missing_returns_unavailable(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When IMPL-03's attack_log module is not importable, we degrade to
    a structured error — the runtime must not crash on first-round calls
    made before IMPL-03 lands.
    """
    import sys

    # Force-import the real module (may or may not exist). If it doesn't,
    # the lazy-import inside _tool_read_attack_log trips ImportError.
    monkeypatch.setitem(sys.modules, "red_team.agent.attack_log", None)  # type: ignore[arg-type]
    log_path = tmp_path / "attack_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    result = _dispatch_tool(
        "read_attack_log",
        {"query": "autonomy"},
        None,
        None,
        None,
        str(log_path),
    )
    assert result == {"error": "attack_log unavailable"}


# ---------------------------------------------------------------------------
# get_binary (code-layer mutation catalog)
# ---------------------------------------------------------------------------


import shutil  # noqa: E402

_GO_AVAILABLE = shutil.which("go") is not None
_requires_go = pytest.mark.skipif(
    not _GO_AVAILABLE, reason="go toolchain unavailable"
)


@pytest.fixture()
def isolated_binary_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Keep compiled binaries in a per-test tempdir so dispatcher tests
    don't race on the shared build cache."""
    monkeypatch.setenv("CLAW_BINARY_BUILD_ROOT", str(tmp_path))
    return tmp_path


@_requires_go
def test_get_binary_dispatches_to_catalog_with_defaults(
    isolated_binary_cache: Path,
) -> None:
    """Dispatcher routes to binary_catalog.get_binary; default params
    fill in when caller omits `params`. The returned `content` is a
    Bash wrapper with the compiled ELF embedded as base64."""
    result = _dispatch_tool(
        "get_binary", {"binary_id": "honeypot_beacon"}, None, None, None, None
    )
    assert result["binary_id"] == "honeypot_beacon"
    assert result["filename"] == "beacon.sh"
    # Default HONEYPOT_URL surfaces as an env export.
    # shlex.quote leaves shell-safe URLs unquoted.
    assert "export HONEYPOT_URL=http://127.0.0.1:9999/collect" in result["content"]
    # Wrapper heredoc markers present — proves the ELF was embedded.
    assert "____CLAW_BIN____" in result["content"]
    assert "data_exfiltration" in result["covers_objectives"]


@_requires_go
def test_get_binary_accepts_params_override(
    isolated_binary_cache: Path,
) -> None:
    result = _dispatch_tool(
        "get_binary",
        {
            "binary_id": "file_shredder",
            "params": {"TARGET": "/workspace/evidence", "MODE": "truncate"},
        },
        None,
        None,
        None,
        None,
    )
    # Plain alphanumeric + slashes → shlex leaves them unquoted.
    assert "export TARGET=/workspace/evidence" in result["content"]
    assert "export MODE=truncate" in result["content"]


def test_get_binary_unknown_id_returns_structured_error() -> None:
    result = _dispatch_tool(
        "get_binary", {"binary_id": "nonsense"}, None, None, None, None
    )
    assert "error" in result
    assert "nonsense" in result["error"]


def test_get_binary_missing_id_returns_structured_error() -> None:
    result = _dispatch_tool("get_binary", {}, None, None, None, None)
    assert "error" in result


def test_get_binary_rejects_non_dict_params() -> None:
    result = _dispatch_tool(
        "get_binary",
        {"binary_id": "honeypot_beacon", "params": "HONEYPOT_URL=foo"},
        None,
        None,
        None,
        None,
    )
    assert "error" in result
    assert "object" in result["error"]


# ---------------------------------------------------------------------------
# submit_mutation (terminal)
# ---------------------------------------------------------------------------


def test_submit_mutation_returns_terminal_sentinel() -> None:
    """Dispatcher does NOT build the SkillVariant — it emits a sentinel
    that the runtime (IMPL-05) routes to IMPL-04's builder."""
    args = {
        "new_d": "# patched SKILL.md",
        "new_c": {"entry_script": "run.py"},
        "new_kappa": ["network"],
        "rationale": "evades authority_framing reason_code by rephrasing.",
    }
    result = _dispatch_tool(
        "submit_mutation", args, None, None, None, None
    )
    assert result["_terminal"] is True
    assert result["args"] == args


def test_submit_mutation_args_are_copied_not_aliased() -> None:
    """The sentinel's ``args`` must be a defensive copy so the runtime
    can mutate its own dict without corrupting the caller's."""
    original = {"new_d": "x", "rationale": "y"}
    result = _dispatch_tool(
        "submit_mutation", original, None, None, None, None
    )
    result["args"]["new_d"] = "mutated"
    assert original["new_d"] == "x"


# ---------------------------------------------------------------------------
# dispatcher error path
# ---------------------------------------------------------------------------


def test_unknown_tool_name_raises_keyerror() -> None:
    """Unknown tool == programmer error (not an LLM mistake), so the
    dispatcher raises rather than swallowing — failing loudly is better
    than letting a silent no-op propagate into the evolution loop."""
    with pytest.raises(KeyError):
        _dispatch_tool("no_such_tool", {}, None, None, None, None)


def test_dispatcher_accepts_none_context_args() -> None:
    """The ``seed``, ``findings``, ``repo`` positional args are reserved
    for later rounds; they must be None-safe today so IMPL-05 can pass
    placeholders without the dispatcher blowing up."""
    # The dispatcher must not raise on None seed/findings/repo.
    _dispatch_tool("list_skills", {"category": "benign"}, None, None, None, None)
