"""Tests for ToolUseAgentStrategy — wraps red_team.agent.runtime.agent_run."""
from __future__ import annotations

from typing import Any

import pytest

from core.models import RepositorySlice
from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.baselines.base import AgentRunResult
from red_team.mutation.base import MutationContext, MutationResult, MutationStrategy


# ---------------------------------------------------------------- fixtures


def _make_variant() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.DATA_EXFILTRATION,
        tau=Topology.SINGLE,
        c={"entry_script": "main.py"},
        kappa=[Channel.FILESYSTEM],
        d="# skill doc",
        chain=["skill-1"],
    )


def _make_repo() -> RepositorySlice:
    return RepositorySlice(
        repo_id="repo-001",
        repo_root="/tmp/repo-001",
        readme_path="README.md",
        readme_content="demo",
        file_tree=["main.py"],
        dependency_files=[],
    )


def _make_ctx(**overrides: Any) -> MutationContext:
    base: dict[str, Any] = dict(
        seed=_make_variant(),
        findings=(),
        repo=_make_repo(),
        search_space="evasion",
    )
    base.update(overrides)
    return MutationContext(**base)


# ---------------------------------------------------------------- tests


def test_tool_use_agent_is_mutation_strategy() -> None:
    from red_team.mutation.strategies.tool_use_agent import ToolUseAgentStrategy

    assert isinstance(ToolUseAgentStrategy(), MutationStrategy)


def test_tool_use_agent_name_is_backend_label() -> None:
    from red_team.mutation.strategies.tool_use_agent import ToolUseAgentStrategy

    strat = ToolUseAgentStrategy()
    assert strat.name == "tool_use_agent"


def test_tool_use_agent_forwards_ctx_fields_to_agent_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrapper must map MutationContext → agent_run kwargs correctly."""
    from red_team.mutation.strategies import tool_use_agent as mod
    from red_team.mutation.strategies.tool_use_agent import ToolUseAgentStrategy

    captured: dict[str, Any] = {}

    def _fake_agent_run(seed, findings, repo, search_space, **kwargs):
        captured["seed"] = seed
        captured["findings"] = findings
        captured["repo"] = repo
        captured["search_space"] = search_space
        captured["kwargs"] = kwargs
        return AgentRunResult(
            mutation=seed,
            tool_chain=["read_skill", "submit_mutation"],
            rationale="done",
            cost_usd=0.25,
            tokens_in=100,
            tokens_out=50,
            refusal=False,
            stop_reason="submitted",
        )

    monkeypatch.setattr(mod, "agent_run", _fake_agent_run)

    ctx = _make_ctx(search_space="evasion", budget_usd=12.5, max_tool_calls=7)
    result = ToolUseAgentStrategy().mutate(ctx)

    # Positional-or-keyword mapping: seed / findings / repo / search_space.
    assert captured["seed"] is ctx.seed
    assert list(captured["findings"]) == list(ctx.findings)
    assert captured["repo"] is ctx.repo
    # search_space may be the semantic name or the S1/S2/S3 translation;
    # either way the underlying SearchSpace accepts it.
    assert captured["search_space"] in {"evasion", "S1"}

    kwargs = captured["kwargs"]
    assert kwargs["budget_usd"] == 12.5
    assert kwargs["max_tool_calls"] == 7

    # Result mapping.
    assert isinstance(result, MutationResult)
    assert result.variant is ctx.seed
    assert result.tool_chain == ("read_skill", "submit_mutation")
    assert result.stop_reason == "submitted"
    assert result.cost_usd == 0.25
    assert result.backend == "tool_use_agent"


def test_tool_use_agent_propagates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """variant=None from agent_run must surface as variant=None in result."""
    from red_team.mutation.strategies import tool_use_agent as mod
    from red_team.mutation.strategies.tool_use_agent import ToolUseAgentStrategy

    def _fake_agent_run(*_, **__):
        return AgentRunResult(
            mutation=None,
            tool_chain=[],
            rationale="no tokens left",
            cost_usd=9.99,
            refusal=False,
            stop_reason="budget",
        )

    monkeypatch.setattr(mod, "agent_run", _fake_agent_run)

    result = ToolUseAgentStrategy().mutate(_make_ctx())
    assert result.variant is None
    assert result.stop_reason == "budget"
    assert result.cost_usd == 9.99
    assert result.backend == "tool_use_agent"


def test_tool_use_agent_maps_all_three_semantic_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each semantic search_space must reach agent_run in a form SearchSpace accepts."""
    from red_team.evolution.search_space import SearchSpace
    from red_team.mutation.strategies import tool_use_agent as mod
    from red_team.mutation.strategies.tool_use_agent import ToolUseAgentStrategy

    seen_spaces: list[str] = []

    def _fake(seed, findings, repo, search_space, **_: Any):
        seen_spaces.append(search_space)
        # Confirm the value is accepted by SearchSpace without a warning.
        SearchSpace(search_space)
        return AgentRunResult(
            mutation=seed,
            tool_chain=[],
            rationale="",
            cost_usd=0.0,
            refusal=False,
            stop_reason="submitted",
        )

    monkeypatch.setattr(mod, "agent_run", _fake)
    strat = ToolUseAgentStrategy()
    for space in ("evasion", "path_expansion", "surface_expansion"):
        strat.mutate(_make_ctx(search_space=space))

    assert len(seen_spaces) == 3
