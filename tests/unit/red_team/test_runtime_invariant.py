"""IMPL-05 — retrieval-first invariant test.

Scenario
--------
Round 1: LLM issues ``submit_mutation`` as its very first tool call.  The
         runtime must push ``{"error": "must call at least one retrieval
         tool first"}`` back as a tool result and keep looping.
Round 2: LLM corrects itself and calls ``read_skill``.  ``retrieval_tool_
         called`` flips to True.
Round 3: LLM retries ``submit_mutation`` — this time it succeeds and
         :func:`agent_run` returns ``stop_reason="submitted"``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.models import RepositorySlice
from core.schemas import AttackObjective, Channel, SkillVariant, Topology


# ---------------------------------------------------------------- fixtures


def _make_seed() -> SkillVariant:
    return SkillVariant(
        # GUIDANCE_INJECTION used so the retrieval-first invariant test
        # does not trigger the v13.4 pre_flight_required check (which
        # would push the loop further than the test mocks expect).
        o=AttackObjective.GUIDANCE_INJECTION,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh", "deps": [], "declared_apis": []},
        kappa=[Channel.FILESYSTEM, Channel.SUBPROCESS],
        d="# original skill doc",
        poison_index=0,
        chain=["skill-alpha"],
        metadata={"seed_id": "seed-test"},
    )


def _make_repo() -> RepositorySlice:
    return RepositorySlice(
        repo_id="slopbuster",
        repo_root="/fake/root",
        readme_path=None,
        readme_content="",
        file_tree=[],
        dependency_files=[],
    )


def _tool_call(call_id: str, name: str, args: dict) -> SimpleNamespace:
    """Build a SimpleNamespace that quacks like an OpenAI tool_call."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        ),
    )


def _llm_response(
    *, tool_calls: list | None = None, content: str | None = None,
    prompt_tokens: int = 10, completion_tokens: int = 5,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(choices=[choice], usage=usage)


# -------------------------------------------------------------------- test


def test_retrieval_first_invariant_retries_until_submit_succeeds(
    monkeypatch,
) -> None:
    """Driver pushes invariant error, loops, and succeeds after retrieval."""
    seed = _make_seed()
    repo = _make_repo()

    # Round 1: LLM tries to submit without any retrieval — must be rejected.
    round1 = _llm_response(
        tool_calls=[
            _tool_call(
                "call-1",
                "submit_mutation",
                {
                    "new_d": "# premature submit",
                    "rationale": "lazy agent",
                },
            )
        ]
    )
    # Round 2: LLM calls retrieval.
    round2 = _llm_response(
        tool_calls=[
            _tool_call(
                "call-2",
                "read_skill",
                {"skill_id": "nonexistent-skill", "split": "benign"},
            )
        ]
    )
    # Round 3: LLM retries submit — now it should succeed.
    round3 = _llm_response(
        tool_calls=[
            _tool_call(
                "call-3",
                "submit_mutation",
                {
                    "new_d": "# polished skill doc",
                    "rationale": "applied reword operator",
                },
            )
        ]
    )

    client = MagicMock()
    client.chat.completions.create.side_effect = [round1, round2, round3]

    # Patch the runtime's client factory rather than ``openai.OpenAI``
    # directly — simpler and avoids depending on the lazy import path.
    with patch(
        "red_team.agent.runtime._make_client", return_value=client
    ):
        from red_team.agent.runtime import agent_run

        result = agent_run(
            seed=seed,
            findings=[],
            repo=repo,
            search_space="evasion",
            max_tool_calls=5,
            budget_usd=10.0,
        )

    # Exactly three LLM turns should have been consumed.
    assert client.chat.completions.create.call_count == 3

    # Final outcome: mutation built, stop_reason submitted.
    assert result.stop_reason == "submitted"
    assert result.mutation is not None
    assert result.refusal is False
    assert result.rationale == "applied reword operator"

    # tool_chain should contain the full trajectory:
    #   (1) rejected submit, (2) read_skill, (3) accepted submit.
    names = [entry["tool"] for entry in result.tool_chain]
    assert names == ["submit_mutation", "read_skill", "submit_mutation"]

    # The first submit's preview must surface the retrieval-first error.
    first_preview = result.tool_chain[0]["result_preview"]
    assert "must call at least one retrieval tool first" in first_preview


def test_invariant_violation_is_surfaced_as_tool_error(monkeypatch) -> None:
    """An evasion mutation that changes c must be fed back as an error, not raised.

    Round 1: retrieval so invariant is satisfied.
    Round 2: submit_mutation carrying an illegal ``new_c`` → runtime
             catches :class:`InvariantViolation` and re-prompts the LLM.
    Round 3: (mock goes silent via plain content) → loop exits cleanly.
    """
    seed = _make_seed()
    repo = _make_repo()

    r1 = _llm_response(
        tool_calls=[
            _tool_call(
                "c-1", "read_skill",
                {"skill_id": "nope", "split": "benign"},
            )
        ]
    )
    r2 = _llm_response(
        tool_calls=[
            _tool_call(
                "c-2",
                "submit_mutation",
                {
                    "new_d": "# attempt",
                    # evasion forbids c changes — must trigger InvariantViolation.
                    "new_c": {"entry_script": "hacked.sh"},
                    "rationale": "whoops",
                },
            )
        ]
    )
    # Round 3: plain text, triggers refusal stop to keep test bounded.
    r3 = _llm_response(
        content="I give up.", tool_calls=None,
    )

    client = MagicMock()
    client.chat.completions.create.side_effect = [r1, r2, r3]

    with patch(
        "red_team.agent.runtime._make_client", return_value=client
    ):
        from red_team.agent.runtime import agent_run

        result = agent_run(
            seed=seed, findings=[], repo=repo, search_space="evasion",
            max_tool_calls=5, budget_usd=10.0,
        )

    # The rejected submit must be logged with an invariant_violation error.
    previews = [e["result_preview"] for e in result.tool_chain]
    assert any("invariant_violation" in p for p in previews)
    # Runtime did NOT terminate with mutation on the violating submit.
    assert result.stop_reason == "refusal"  # terminated on round 3 plain text
    assert result.mutation is None
