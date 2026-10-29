"""IMPL-05 — refusal handling test.

When the LLM returns plain text (no ``tool_calls``) the runtime must
surface ``refusal=True`` with ``stop_reason="refusal"`` and preserve
the model's message as ``rationale`` so the caller can log the reason.
"""
from __future__ import annotations

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


def _plain_text_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_plain_text_response_produces_refusal() -> None:
    seed = _make_seed()
    repo = _make_repo()

    client = MagicMock()
    client.chat.completions.create.return_value = _plain_text_response(
        "I decline to help with crafting offensive skills."
    )

    with patch(
        "red_team.agent.runtime._make_client", return_value=client
    ):
        from red_team.agent.runtime import agent_run

        result = agent_run(
            seed=seed, findings=[], repo=repo, search_space="evasion",
            max_tool_calls=5, budget_usd=10.0,
        )

    # Loop stopped on the first turn.
    assert client.chat.completions.create.call_count == 1

    assert result.stop_reason == "refusal"
    assert result.refusal is True
    assert result.mutation is None
    # Rationale preserves the model's explanation so logs are useful.
    assert "decline" in result.rationale.lower()
    # Usage was still accounted for — refusals burn tokens too.
    assert result.tokens_in == 7
    assert result.tokens_out == 3


def test_empty_tool_calls_list_also_counts_as_refusal() -> None:
    """Some providers send ``tool_calls=[]`` instead of ``None``."""
    seed = _make_seed()
    repo = _make_repo()

    message = SimpleNamespace(
        content="Not today.", tool_calls=[]
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
    )

    client = MagicMock()
    client.chat.completions.create.return_value = response

    with patch(
        "red_team.agent.runtime._make_client", return_value=client
    ):
        from red_team.agent.runtime import agent_run

        result = agent_run(
            seed=seed, findings=[], repo=repo, search_space="evasion",
            max_tool_calls=5, budget_usd=10.0,
        )

    assert result.stop_reason == "refusal"
    assert result.refusal is True
    assert result.mutation is None
    assert result.rationale == "Not today."
