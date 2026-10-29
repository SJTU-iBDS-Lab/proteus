"""Tests for RandomStrategy — wraps red_team.baselines.random_mutation.RandomMutation."""
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


def test_random_is_mutation_strategy() -> None:
    from red_team.mutation.strategies.random import RandomStrategy

    assert isinstance(RandomStrategy(), MutationStrategy)


def test_random_name_is_backend_label() -> None:
    from red_team.mutation.strategies.random import RandomStrategy

    assert RandomStrategy().name == "random"


def test_random_strategy_forwards_to_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """RandomStrategy must call RandomMutation.agent_run with a legacy level.

    The baseline's ``_build_candidate`` branches on ``level == "S2"`` /
    ``"S3"``; if we passed the semantic name through the strategy would
    silently drop the level-specific ops.  So the wrapper must translate
    ``evasion``/``path_expansion``/``surface_expansion`` → ``S1``/``S2``/``S3``.
    """
    from red_team.mutation.strategies import random as mod
    from red_team.mutation.strategies.random import RandomStrategy

    captured: dict[str, Any] = {}

    class _FakeBaseline:
        name = "random"

        def agent_run(self, seed, findings, repo, search_space, **kwargs):
            captured["seed"] = seed
            captured["findings"] = findings
            captured["repo"] = repo
            captured["search_space"] = search_space
            captured["kwargs"] = kwargs
            return AgentRunResult(
                mutation=seed,
                tool_chain=[],
                rationale="S2:append",
                cost_usd=0.0,
                refusal=False,
                stop_reason="random",
            )

    monkeypatch.setattr(mod, "RandomMutation", _FakeBaseline)

    ctx = _make_ctx(search_space="path_expansion", extra={"rng_seed": 7})
    result = RandomStrategy().mutate(ctx)

    assert captured["seed"] is ctx.seed
    assert captured["search_space"] == "S2"
    assert captured["kwargs"].get("rng_seed") == 7

    assert isinstance(result, MutationResult)
    assert result.variant is ctx.seed
    assert result.stop_reason == "random"
    assert result.cost_usd == 0.0
    assert result.backend == "random"
    assert result.tool_chain == ()


def test_random_strategy_translates_all_three_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from red_team.mutation.strategies import random as mod
    from red_team.mutation.strategies.random import RandomStrategy

    seen: list[str] = []

    class _FakeBaseline:
        name = "random"

        def agent_run(self, seed, findings, repo, search_space, **_: Any):
            seen.append(search_space)
            return AgentRunResult(
                mutation=seed,
                tool_chain=[],
                rationale="",
                cost_usd=0.0,
                refusal=False,
                stop_reason="random",
            )

    monkeypatch.setattr(mod, "RandomMutation", _FakeBaseline)

    strat = RandomStrategy()
    strat.mutate(_make_ctx(search_space="evasion"))
    strat.mutate(_make_ctx(search_space="path_expansion"))
    strat.mutate(_make_ctx(search_space="surface_expansion"))

    assert seen == ["S1", "S2", "S3"]


def test_random_strategy_propagates_invariant_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from red_team.mutation.strategies import random as mod
    from red_team.mutation.strategies.random import RandomStrategy

    class _FakeBaseline:
        name = "random"

        def agent_run(self, *_, **__):
            return AgentRunResult(
                mutation=None,
                tool_chain=[],
                rationale="random_invariant_failed: boom",
                cost_usd=0.0,
                refusal=False,
                stop_reason="invariant_violation",
            )

    monkeypatch.setattr(mod, "RandomMutation", _FakeBaseline)

    result = RandomStrategy().mutate(_make_ctx())
    assert result.variant is None
    assert result.stop_reason == "invariant_violation"
    assert result.backend == "random"
