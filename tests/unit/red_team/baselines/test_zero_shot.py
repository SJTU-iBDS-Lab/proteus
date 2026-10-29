"""Unit tests for the LLM zero-shot baseline (IMPL-17, T3).

All tests mock ``openai.OpenAI`` so no network traffic ever hits the real
LiteLLM proxy.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.baselines.base import estimate_cost
from red_team.baselines.llm_zeroshot import LLMZeroShot
from red_team.skill_poisoner import build_poisoned_prompt


# --------------------------------------------------------------------- helpers


def _make_seed() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.DATA_EXFILTRATION,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=[Channel.SUBPROCESS],
        d="# original skill\nbenign-sounding body",
        poison_index=0,
        chain=["skill-a"],
        metadata={"seed_id": "seed-17"},
    )


def _make_openai_response(
    *,
    content: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> SimpleNamespace:
    """Build a minimal object that quacks like an OpenAI ChatCompletion."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = None
    if prompt_tokens is not None or completion_tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
        )
    return SimpleNamespace(choices=[choice], usage=usage)


def _mock_client_returning(response: SimpleNamespace) -> MagicMock:
    """Return a MagicMock that looks like an openai.OpenAI instance."""
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------- tests


def test_name_attribute() -> None:
    assert LLMZeroShot.name == "llm_zeroshot"
    assert LLMZeroShot().name == "llm_zeroshot"


def test_agent_run_uses_llm_rewrite() -> None:
    seed = _make_seed()
    rewritten = "# mutated skill body from llm\nevade all scanners"
    response = _make_openai_response(
        content=rewritten,
        prompt_tokens=123,
        completion_tokens=45,
    )
    client = _mock_client_returning(response)

    with (
        patch("openai.OpenAI", return_value=client) as mock_ctor,
        patch(
            "red_team.baselines.llm_zeroshot.make_variant_from_new_d",
            wraps=__import__(
                "red_team.baselines.llm_zeroshot",
                fromlist=["make_variant_from_new_d"],
            ).make_variant_from_new_d,
        ) as mock_make,
    ):
        result = LLMZeroShot().agent_run(
            seed, findings=[], repo=None, search_space="",
        )

    # Client was constructed against the configured endpoint.
    mock_ctor.assert_called_once()
    kwargs = mock_ctor.call_args.kwargs
    assert kwargs["base_url"]  # non-empty
    assert kwargs["api_key"]   # non-empty

    # Exactly one chat.completions.create call, with model=glm-5 by default.
    client.chat.completions.create.assert_called_once()
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "glm-5"
    messages = call_kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    # Prompt contains the objective + current body.
    assert "data_exfiltration" in messages[0]["content"]
    assert "benign-sounding body" in messages[0]["content"]

    # make_variant_from_new_d was called with the LLM-produced rewrite.
    mock_make.assert_called_once()
    assert mock_make.call_args.args[1] == rewritten

    # Result wraps the mutation with the expected contract.
    assert result.mutation is not None
    assert rewritten in result.mutation.d
    assert result.mutation.o == seed.o
    assert result.tool_chain == []
    assert result.refusal is False
    assert result.stop_reason == "zero_shot"
    assert result.tokens_in == 123
    assert result.tokens_out == 45


def test_agent_run_empty_response_falls_back_to_poison_prompt() -> None:
    seed = _make_seed()
    response = _make_openai_response(
        content="   \n  ",  # whitespace-only → triggers fallback
        prompt_tokens=10,
        completion_tokens=0,
    )
    client = _mock_client_returning(response)

    with patch("openai.OpenAI", return_value=client):
        result = LLMZeroShot().agent_run(
            seed, findings=[], repo=None, search_space="",
        )

    expected_body = seed.d + "\n\n" + build_poisoned_prompt(seed.o.value)
    assert result.mutation is not None
    assert result.mutation.d == expected_body
    assert result.tool_chain == []
    assert result.refusal is False
    assert result.stop_reason == "zero_shot"


def test_agent_run_cost_matches_estimate_cost() -> None:
    seed = _make_seed()
    response = _make_openai_response(
        content="new body",
        prompt_tokens=100,
        completion_tokens=50,
    )
    client = _mock_client_returning(response)

    with patch("openai.OpenAI", return_value=client):
        result = LLMZeroShot().agent_run(
            seed, findings=[], repo=None, search_space="",
        )

    assert result.tokens_in == 100
    assert result.tokens_out == 50
    assert result.cost_usd == pytest.approx(estimate_cost(100, 50, "glm-5"))
    assert result.cost_usd > 0.0


def test_agent_run_tool_chain_and_refusal_invariants() -> None:
    """Across several mock outcomes, tool_chain is always empty and
    refusal is always False — T3 never retries and never refuses."""
    seed = _make_seed()
    outcomes = [
        _make_openai_response(content="valid rewrite", prompt_tokens=5, completion_tokens=5),
        _make_openai_response(content="", prompt_tokens=5, completion_tokens=0),
        _make_openai_response(content="another rewrite", prompt_tokens=0, completion_tokens=0),
    ]
    for resp in outcomes:
        client = _mock_client_returning(resp)
        with patch("openai.OpenAI", return_value=client):
            result = LLMZeroShot().agent_run(
                seed, findings=[], repo=None, search_space="",
            )
        assert result.tool_chain == []
        assert result.refusal is False
        assert result.stop_reason == "zero_shot"


def test_agent_run_honours_model_override() -> None:
    seed = _make_seed()
    response = _make_openai_response(
        content="mutated",
        prompt_tokens=200,
        completion_tokens=100,
    )
    client = _mock_client_returning(response)

    with patch("openai.OpenAI", return_value=client):
        result = LLMZeroShot().agent_run(
            seed, findings=[], repo=None, search_space="",
            model="gpt-5.4",
        )

    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-5.4"
    # Cost must reflect the overridden model's rate card.
    assert result.cost_usd == pytest.approx(estimate_cost(200, 100, "gpt-5.4"))


def test_agent_run_uses_litellm_url_and_api_key_overrides() -> None:
    seed = _make_seed()
    response = _make_openai_response(content="x", prompt_tokens=1, completion_tokens=1)
    client = _mock_client_returning(response)

    with patch("openai.OpenAI", return_value=client) as mock_ctor:
        LLMZeroShot().agent_run(
            seed, findings=[], repo=None, search_space="",
            litellm_url="http://example.test:9999",
            api_key="sk-test-override",
        )

    kwargs = mock_ctor.call_args.kwargs
    assert kwargs["base_url"] == "http://example.test:9999"
    assert kwargs["api_key"] == "sk-test-override"


def test_agent_run_ignores_findings_by_design() -> None:
    """T3 must not observe or propagate blue-team findings."""
    seed = _make_seed()
    response = _make_openai_response(content="body", prompt_tokens=1, completion_tokens=1)
    client = _mock_client_returning(response)

    # Pass some non-empty findings list; baseline must not mention it in
    # the outgoing prompt.
    findings = [object(), object()]  # opaque sentinels — would explode if read
    with patch("openai.OpenAI", return_value=client):
        result = LLMZeroShot().agent_run(
            seed, findings=findings, repo=None, search_space="",
        )

    call_kwargs = client.chat.completions.create.call_args.kwargs
    prompt_text = call_kwargs["messages"][0]["content"]
    assert "finding" not in prompt_text.lower()
    assert result.stop_reason == "zero_shot"
