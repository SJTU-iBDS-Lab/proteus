"""Tests for BlackboxStrategy — wraps red_team.baselines.blackbox_rewrite.BlackboxRewrite."""
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


def test_blackbox_is_mutation_strategy() -> None:
    from red_team.mutation.strategies.blackbox import BlackboxStrategy

    assert isinstance(BlackboxStrategy(), MutationStrategy)


def test_blackbox_name_is_backend_label() -> None:
    from red_team.mutation.strategies.blackbox import BlackboxStrategy

    assert BlackboxStrategy().name == "blackbox"


def test_blackbox_forwards_round_index_and_target_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BlackboxRewrite.agent_run requires ``round_index``; pull from ctx.extra."""
    from red_team.mutation.strategies import blackbox as mod
    from red_team.mutation.strategies.blackbox import BlackboxStrategy

    captured: dict[str, Any] = {}

    class _FakeBaseline:
        name = "blackbox"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["ctor_args"] = args
            captured["ctor_kwargs"] = kwargs

        def agent_run(self, seed, findings, repo, search_space, **kwargs):
            captured["seed"] = seed
            captured["findings"] = findings
            captured["search_space"] = search_space
            captured["kwargs"] = kwargs
            return AgentRunResult(
                mutation=seed,
                tool_chain=[],
                rationale="blackbox rewrite @ round 3",
                cost_usd=0.07,
                tokens_in=40,
                tokens_out=30,
                refusal=False,
                stop_reason="blackbox",
            )

    monkeypatch.setattr(mod, "BlackboxRewrite", _FakeBaseline)

    ctx = _make_ctx(
        target_model="glm-5",
        extra={"round_index": 3},
    )
    result = BlackboxStrategy().mutate(ctx)

    assert captured["kwargs"].get("round_index") == 3
    assert captured["kwargs"].get("model") == "glm-5"

    assert isinstance(result, MutationResult)
    assert result.variant is ctx.seed
    assert result.stop_reason == "blackbox"
    assert result.cost_usd == 0.07
    assert result.backend == "blackbox"
    assert result.tool_chain == ()


def test_blackbox_defaults_round_index_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent round_index must default to 0 so the baseline never crashes."""
    from red_team.mutation.strategies import blackbox as mod
    from red_team.mutation.strategies.blackbox import BlackboxStrategy

    captured: dict[str, Any] = {}

    class _FakeBaseline:
        name = "blackbox"

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def agent_run(self, *_, **kwargs):
            captured["kwargs"] = kwargs
            return AgentRunResult(
                mutation=_make_variant(),
                tool_chain=[],
                rationale="ok",
                cost_usd=0.0,
                refusal=False,
                stop_reason="blackbox",
            )

    monkeypatch.setattr(mod, "BlackboxRewrite", _FakeBaseline)

    BlackboxStrategy().mutate(_make_ctx())
    assert captured["kwargs"].get("round_index") == 0


def test_blackbox_propagates_empty_rewrite_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from red_team.mutation.strategies import blackbox as mod
    from red_team.mutation.strategies.blackbox import BlackboxStrategy

    class _FakeBaseline:
        name = "blackbox"

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def agent_run(self, *_, **__):
            return AgentRunResult(
                mutation=None,
                tool_chain=[],
                rationale="empty rewrite",
                cost_usd=0.005,
                tokens_in=10,
                tokens_out=0,
                refusal=False,
                stop_reason="blackbox",
            )

    monkeypatch.setattr(mod, "BlackboxRewrite", _FakeBaseline)

    result = BlackboxStrategy().mutate(_make_ctx(extra={"round_index": 1}))
    assert result.variant is None
    assert result.stop_reason == "blackbox"
    assert result.backend == "blackbox"
