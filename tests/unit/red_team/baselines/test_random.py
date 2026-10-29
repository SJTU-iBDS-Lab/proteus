"""Unit tests for the LLM-free random-mutation baseline (IMPL-16)."""
from __future__ import annotations

from typing import Any

import pytest

from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.baselines.base import AgentRunResult
from red_team.baselines.random_mutation import RandomMutation
from red_team.evolution.search_space import SearchSpace


# ---------------------------------------------------------------- fixtures

def _make_parent() -> SkillVariant:
    """A modest S1-ready seed with ## headers so replace-section has work."""
    body = (
        "# Skill\n\n"
        "## Overview\n"
        "This skill summarizes docs.\n\n"
        "## Usage\n"
        "Run it with a filename.\n"
    )
    return SkillVariant(
        o=AttackObjective.DATA_EXFILTRATION,
        tau=Topology.SINGLE,
        c={
            "entry_script": "run.sh",
            "scripts": {"run.sh": "#!/usr/bin/env bash\necho hello\n"},
            "deps": ["curl"],
        },
        kappa=[Channel.SUBPROCESS, Channel.NETWORK],
        d=body,
        poison_index=0,
        chain=["skill-a"],
        metadata={"seed_id": "seed-random"},
    )


class _StubRepo:
    """Minimal RepositorySlice stand-in; the baseline does not inspect it."""

    repo_id: str = "stub"
    repo_root: str = "/tmp"
    readme_path: str | None = None
    readme_content: str = ""
    file_tree: list[str] = []
    dependency_files: list[str] = []
    labels: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    extracted_themes: list[str] = []


# ----------------------------------------------------------------- S1 suite

def test_s1_ten_runs_produce_valid_mutations() -> None:
    parent = _make_parent()
    repo = _StubRepo()
    baseline = RandomMutation()
    space = SearchSpace("evasion")

    results: list[AgentRunResult] = []
    for seed in range(10):
        r = baseline.agent_run(parent, [], repo, "evasion", rng_seed=seed)
        results.append(r)

    assert len(results) == 10
    for r in results:
        assert r.mutation is not None, r.rationale
        assert r.tool_chain == []
        assert r.stop_reason == "random"
        assert r.cost_usd == 0.0
        assert r.tokens_in == 0
        assert r.tokens_out == 0
        assert r.refusal is False
        # Critical: validates under S1 (only d mutated).
        space.validate(parent, r.mutation)
        # d must have actually changed.
        assert r.mutation.d != parent.d


def test_rng_seed_reproducibility() -> None:
    parent = _make_parent()
    repo = _StubRepo()
    baseline = RandomMutation()

    r1 = baseline.agent_run(parent, [], repo, "evasion", rng_seed=42)
    r2 = baseline.agent_run(parent, [], repo, "evasion", rng_seed=42)

    assert r1.mutation is not None and r2.mutation is not None
    assert r1.mutation.d == r2.mutation.d
    assert r1.mutation.content_hash() == r2.mutation.content_hash()


def test_different_seeds_give_different_outputs() -> None:
    # Not strictly part of the spec but a useful smoke test — two different
    # seeds should produce different d texts with overwhelming probability.
    parent = _make_parent()
    repo = _StubRepo()
    baseline = RandomMutation()

    r1 = baseline.agent_run(parent, [], repo, "evasion", rng_seed=1)
    r2 = baseline.agent_run(parent, [], repo, "evasion", rng_seed=2)
    assert r1.mutation is not None and r2.mutation is not None
    assert r1.mutation.d != r2.mutation.d


# ----------------------------------------------------------------- S2 suite

def test_s2_output_validates_under_s2() -> None:
    parent = _make_parent()
    repo = _StubRepo()
    baseline = RandomMutation()
    space = SearchSpace("path_expansion")

    # Run multiple seeds so we cover both the "mutate c" and "leave c" paths
    # of the P=0.5 branch.
    for seed in range(20):
        r = baseline.agent_run(parent, [], repo, "path_expansion", rng_seed=seed)
        assert r.mutation is not None
        assert r.tool_chain == []
        assert r.stop_reason == "random"
        # Validates: o and tau must not change; d/c/chain/kappa may.
        space.validate(parent, r.mutation)


# ----------------------------------------------------------------- S3 suite

def test_s3_rotates_objective_and_shares_motif() -> None:
    parent = _make_parent()
    repo = _StubRepo()
    baseline = RandomMutation()
    space = SearchSpace("surface_expansion")

    for seed in range(10):
        r = baseline.agent_run(parent, [], repo, "surface_expansion", rng_seed=seed)
        assert r.mutation is not None
        assert r.mutation.o != parent.o
        # Shared-motif fallback relies on κ overlap + entry_script equality.
        assert set(r.mutation.kappa) & set(parent.kappa)
        assert r.mutation.c.get("entry_script") == parent.c.get("entry_script")
        space.validate(parent, r.mutation)


# ------------------------------------------------------------ contract bits

def test_tool_chain_always_empty() -> None:
    parent = _make_parent()
    repo = _StubRepo()
    baseline = RandomMutation()
    for level in ("evasion", "path_expansion", "surface_expansion"):
        r = baseline.agent_run(parent, [], repo, level, rng_seed=7)
        assert r.tool_chain == []


def test_invalid_level_raises() -> None:
    parent = _make_parent()
    repo = _StubRepo()
    baseline = RandomMutation()
    with pytest.raises(ValueError):
        baseline.agent_run(parent, [], repo, "S9", rng_seed=0)


def test_rng_seed_none_still_returns_valid_mutation() -> None:
    # No seed → os.urandom path; just smoke-test that it produces a valid evasion.
    parent = _make_parent()
    repo = _StubRepo()
    baseline = RandomMutation()
    r = baseline.agent_run(parent, [], repo, "evasion")
    assert r.mutation is not None
    SearchSpace("evasion").validate(parent, r.mutation)


def test_agent_run_result_fields_match_contract() -> None:
    parent = _make_parent()
    repo = _StubRepo()
    baseline = RandomMutation()
    r = baseline.agent_run(parent, [], repo, "evasion", rng_seed=3)
    # All IMPL-05 fields are present on the dataclass.
    assert hasattr(r, "mutation")
    assert hasattr(r, "tool_chain")
    assert hasattr(r, "rationale")
    assert hasattr(r, "cost_usd")
    assert hasattr(r, "tokens_in")
    assert hasattr(r, "tokens_out")
    assert hasattr(r, "refusal")
    assert hasattr(r, "stop_reason")
