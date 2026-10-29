"""IMPL-05 — budget enforcement test.

Set ``budget_usd`` near zero and return a mocked usage large enough to
blow through it on the first turn.  The runtime must stop with
``stop_reason="budget"`` on the *next* iteration without calling the
LLM again.
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


def _make_seed() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh", "deps": [], "declared_apis": []},
        kappa=[Channel.FILESYSTEM, Channel.SUBPROCESS],
        d="# original",
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
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name, arguments=json.dumps(args)
        ),
    )


def _huge_cost_response() -> SimpleNamespace:
    # 1_000_000 prompt tokens easily blows a $0.001 budget with the
    # claude-sonnet-4-6 rate card ($0.003/1k in × 1.3 multiplier ≈ $3.9).
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            _tool_call(
                "call-budget",
                "read_skill",
                {"skill_id": "x", "split": "benign"},
            )
        ],
    )
    usage = SimpleNamespace(
        prompt_tokens=1_000_000, completion_tokens=0
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=usage
    )


def test_budget_exceeded_stops_loop_after_first_turn() -> None:
    seed = _make_seed()
    repo = _make_repo()

    client = MagicMock()
    # Even if the mock gets called many times, returning the same huge
    # response guarantees cost stays well over budget.
    client.chat.completions.create.return_value = _huge_cost_response()

    with patch(
        "red_team.agent.runtime._make_client", return_value=client
    ):
        from red_team.agent.runtime import agent_run

        result = agent_run(
            seed=seed, findings=[], repo=repo, search_space="evasion",
            budget_usd=0.001,  # basically zero
            max_tool_calls=5,
            mutator_model="claude-sonnet-4-6",
        )

    # The loop must have consumed exactly ONE LLM call before the
    # budget check at the top of the next iteration fired.
    assert client.chat.completions.create.call_count == 1

    assert result.stop_reason == "budget"
    assert result.mutation is None
    assert result.refusal is False
    # Cost was accumulated from that one oversize call — must be > budget.
    assert result.cost_usd > 0.001
    # Tokens were also accumulated.
    assert result.tokens_in == 1_000_000


def test_budget_not_triggered_on_cheap_call() -> None:
    """Sanity check: a cheap call must NOT trigger the budget path.

    This guards against false positives — if budget logic misfired,
    the test suite would still pass the "hits budget" case while any
    real run would wrongly stop.
    """
    seed = _make_seed()
    repo = _make_repo()

    # Tiny usage: 1 prompt + 1 completion token.
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            _tool_call(
                "cheap", "read_skill",
                {"skill_id": "nope", "split": "benign"},
            )
        ],
    )
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=usage
    )

    # Second turn: refusal (plain text) so the test terminates quickly.
    refusal_msg = SimpleNamespace(content="stopping here", tool_calls=None)
    refusal_resp = SimpleNamespace(
        choices=[SimpleNamespace(message=refusal_msg)],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    client = MagicMock()
    client.chat.completions.create.side_effect = [response, refusal_resp]

    with patch(
        "red_team.agent.runtime._make_client", return_value=client
    ):
        from red_team.agent.runtime import agent_run

        result = agent_run(
            seed=seed, findings=[], repo=repo, search_space="evasion",
            budget_usd=10.0,  # way above real cost
            max_tool_calls=5,
        )

    # Terminated by refusal, not budget.
    assert result.stop_reason == "refusal"
    assert result.cost_usd < 10.0
