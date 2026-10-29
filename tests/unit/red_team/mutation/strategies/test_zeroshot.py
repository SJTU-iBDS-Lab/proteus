"""Tests for ZeroshotStrategy — wraps red_team.baselines.llm_zeroshot.LLMZeroShot."""
from __future__ import annotations

from typing import Any

import pytest

from core.models import RepositorySlice
from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.baselines.base import AgentRunResult
from red_team.mutation.base import MutationContext, MutationResult, MutationStrategy


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


def test_zeroshot_is_mutation_strategy() -> None:
    from red_team.mutation.strategies.zeroshot import ZeroshotStrategy

    assert isinstance(ZeroshotStrategy(), MutationStrategy)


def test_zeroshot_name_is_backend_label() -> None:
    from red_team.mutation.strategies.zeroshot import ZeroshotStrategy

    assert ZeroshotStrategy().name == "zeroshot"


def test_zeroshot_forwards_target_model_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strategy must forward seed + target_model into LLMZeroShot.agent_run."""
    from red_team.mutation.strategies import zeroshot as mod
    from red_team.mutation.strategies.zeroshot import ZeroshotStrategy

    captured: dict[str, Any] = {}

    class _FakeBaseline:
        name = "llm_zeroshot"

        def agent_run(self, seed, findings, repo, search_space, **kwargs):
            captured["seed"] = seed
            captured["findings"] = findings
            captured["repo"] = repo
            captured["search_space"] = search_space
            captured["kwargs"] = kwargs
            return AgentRunResult(
                mutation=seed,
                tool_chain=[],
                rationale="zero-shot single-call SKILL.md rewrite",
                cost_usd=0.12,
                tokens_in=50,
                tokens_out=20,
                refusal=False,
                stop_reason="zero_shot",
            )

    monkeypatch.setattr(mod, "LLMZeroShot", _FakeBaseline)

    ctx = _make_ctx(target_model="kimi-k2.5")
    result = ZeroshotStrategy().mutate(ctx)

    assert captured["seed"] is ctx.seed
    assert captured["kwargs"].get("model") == "kimi-k2.5"

    assert isinstance(result, MutationResult)
    assert result.variant is ctx.seed
    assert result.stop_reason == "zero_shot"
    assert result.cost_usd == 0.12
    assert result.backend == "zeroshot"
    assert result.tool_chain == ()


def test_zeroshot_propagates_refusal_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from red_team.mutation.strategies import zeroshot as mod
    from red_team.mutation.strategies.zeroshot import ZeroshotStrategy

    class _FakeBaseline:
        name = "llm_zeroshot"

        def agent_run(self, *_, **__):
            return AgentRunResult(
                mutation=None,
                tool_chain=[],
                rationale="refusal",
                cost_usd=0.01,
                refusal=True,
                stop_reason="refusal",
            )

    monkeypatch.setattr(mod, "LLMZeroShot", _FakeBaseline)

    result = ZeroshotStrategy().mutate(_make_ctx())
    assert result.variant is None
    assert result.stop_reason == "refusal"
    assert result.backend == "zeroshot"
