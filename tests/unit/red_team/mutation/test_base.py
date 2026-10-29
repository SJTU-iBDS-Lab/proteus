"""Base interface tests for ``red_team.mutation.base``.

The base module defines three things that future mutation strategies
must respect:

* :class:`MutationContext`  — frozen, read-only input package; validates
  ``search_space`` against the semantic vocabulary (``evasion`` /
  ``path_expansion`` / ``surface_expansion``) and rejects the legacy
  ``S1`` / ``S2`` / ``S3`` positional codes.
* :class:`MutationResult`   — frozen output package; ``variant=None`` is a
  legal "strategy gave up" signal; ``backend`` stays a free-form string
  so new strategies can be added without touching the base module.
* :class:`MutationStrategy` — ``runtime_checkable`` Protocol. A class
  must expose a ``name`` attribute and a ``mutate(ctx)`` method to pass
  :func:`isinstance` — forgetting either breaks the check.
"""
from __future__ import annotations

import dataclasses

import pytest

from core.models import RepositorySlice
from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.mutation.base import (
    MutationContext,
    MutationResult,
    MutationStrategy,
)


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


def _make_ctx(**overrides) -> MutationContext:
    base = dict(
        seed=_make_variant(),
        findings=(),
        repo=_make_repo(),
        search_space="evasion",
    )
    base.update(overrides)
    return MutationContext(**base)


# ----------------------------------------------------------- MutationContext


def test_mutation_context_accepts_valid_defaults() -> None:
    ctx = _make_ctx()
    assert ctx.search_space == "evasion"
    assert ctx.findings == ()
    assert ctx.budget_usd == 50.0
    assert ctx.max_tool_calls == 10
    assert ctx.target_model == "glm-5"
    assert ctx.extra == {}


@pytest.mark.parametrize(
    "space",
    ["evasion", "path_expansion", "surface_expansion"],
)
def test_mutation_context_accepts_all_semantic_search_spaces(space: str) -> None:
    ctx = _make_ctx(search_space=space)
    assert ctx.search_space == space


@pytest.mark.parametrize(
    "bad_space",
    ["S1", "S2", "S3", "s1", "", "EVASION", "unknown", "random"],
)
def test_mutation_context_rejects_non_semantic_search_space(bad_space: str) -> None:
    with pytest.raises(ValueError):
        _make_ctx(search_space=bad_space)


def test_mutation_context_valid_spaces_exactly_three() -> None:
    valid = MutationContext._VALID_SPACES
    assert valid == frozenset({"evasion", "path_expansion", "surface_expansion"})
    assert "S1" not in valid
    assert "S2" not in valid
    assert "S3" not in valid


def test_mutation_context_is_frozen() -> None:
    ctx = _make_ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.search_space = "path_expansion"  # type: ignore[misc]


def test_mutation_context_extra_defaults_independent() -> None:
    a = _make_ctx()
    b = _make_ctx()
    # Must not share the default dict instance across constructions.
    assert a.extra is not b.extra


# ----------------------------------------------------------- MutationResult


def test_mutation_result_accepts_valid_variant() -> None:
    variant = _make_variant()
    result = MutationResult(
        variant=variant,
        tool_chain=("search_skill",),
        stop_reason="submitted",
        cost_usd=0.42,
        backend="tool_use_agent",
    )
    assert result.variant is variant
    assert result.tool_chain == ("search_skill",)
    assert result.stop_reason == "submitted"
    assert result.backend == "tool_use_agent"


def test_mutation_result_allows_variant_none_for_failed_strategy() -> None:
    result = MutationResult(
        variant=None,
        tool_chain=(),
        stop_reason="refusal",
        cost_usd=0.0,
        backend="tool_use_agent",
    )
    assert result.variant is None


def test_mutation_result_is_frozen() -> None:
    result = MutationResult(
        variant=None,
        tool_chain=(),
        stop_reason="budget",
        cost_usd=1.0,
        backend="random",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.cost_usd = 2.0  # type: ignore[misc]


def test_mutation_result_backend_is_free_form_string() -> None:
    # No enum guard — future strategies must be addable without editing base.
    result = MutationResult(
        variant=None,
        tool_chain=(),
        stop_reason="programmatic",
        cost_usd=0.0,
        backend="future_tree_search",
    )
    assert isinstance(result.backend, str)
    assert result.backend == "future_tree_search"


def test_mutation_result_tool_chain_is_tuple() -> None:
    result = MutationResult(
        variant=_make_variant(),
        tool_chain=("a", "b", "c"),
        stop_reason="submitted",
        cost_usd=0.0,
        backend="zeroshot",
    )
    assert isinstance(result.tool_chain, tuple)


# -------------------------------------------------------- MutationStrategy


class _DummyStrategy:
    """Inline fake implementing the full protocol."""

    name = "dummy"

    def mutate(self, ctx: MutationContext) -> MutationResult:
        return MutationResult(
            variant=ctx.seed,
            tool_chain=(),
            stop_reason="submitted",
            cost_usd=0.0,
            backend=self.name,
        )


class _IncompleteStrategy:
    """Only has ``name`` — no ``mutate`` — must fail the protocol check."""

    name = "incomplete"


def test_mutation_strategy_accepts_complete_dummy_via_isinstance() -> None:
    assert isinstance(_DummyStrategy(), MutationStrategy)


def test_mutation_strategy_rejects_missing_mutate() -> None:
    assert not isinstance(_IncompleteStrategy(), MutationStrategy)


def test_mutation_strategy_dummy_actually_runs_end_to_end() -> None:
    strategy: MutationStrategy = _DummyStrategy()
    ctx = _make_ctx()
    result = strategy.mutate(ctx)
    assert result.variant is ctx.seed
    assert result.backend == "dummy"
    assert result.stop_reason == "submitted"
