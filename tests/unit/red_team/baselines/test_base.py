"""Unit tests for red-team baseline helpers (IMPL-15)."""
from __future__ import annotations

import random

import pytest

from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.baselines.base import (
    COST_PER_1K,
    estimate_cost,
    make_variant_from_new_d,
    sample_payload_fragment,
)


# ---------------------------------------------------------------------- cost

def test_cost_table_contains_required_models() -> None:
    # All five target models that appear in Table 3/4/5 must be priced.
    for model in ("glm-5", "gpt-5.4", "kimi-k2.5", "claude-sonnet-4-6", "minimax-m1"):
        assert model in COST_PER_1K
        assert set(COST_PER_1K[model].keys()) == {"in", "out"}


def test_estimate_cost_glm5() -> None:
    # 1000 in + 500 out @ glm-5 (0.003 / 0.009 per 1k) with 1.3x safety margin.
    expected = (0.003 + 0.5 * 0.009) * 1.3
    assert estimate_cost(1000, 500, "glm-5") == pytest.approx(expected)


def test_estimate_cost_normalizes_model_name() -> None:
    # Upper-case + surrounding whitespace should still resolve to glm-5.
    expected = (0.003 + 0.5 * 0.009) * 1.3
    assert estimate_cost(1000, 500, "  GLM-5 ") == pytest.approx(expected)


def test_estimate_cost_unknown_model_raises() -> None:
    # No silent fallback — unknown model name fails loud so a typo or
    # forgotten default doesn't silently mis-count cost.
    with pytest.raises(KeyError, match="unknown model id"):
        estimate_cost(1000, 500, "unknown-model-xyz")


def test_estimate_cost_zero_tokens() -> None:
    assert estimate_cost(0, 0, "glm-5") == 0.0


# ------------------------------------------------------------------- variant

def _make_parent() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.DATA_EXFILTRATION,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh", "deps": ["curl"]},
        kappa=[Channel.SUBPROCESS, Channel.NETWORK],
        d="original SKILL.md body",
        poison_index=0,
        chain=["skill-a"],
        metadata={"seed_id": "seed-04"},
    )


def test_make_variant_from_new_d_preserves_fields() -> None:
    parent = _make_parent()
    new_d = "mutated SKILL.md body"
    child = make_variant_from_new_d(parent, new_d)

    assert child.d == new_d
    assert child.o == parent.o
    assert child.tau == parent.tau
    assert child.c == parent.c
    assert child.kappa == parent.kappa
    assert child.poison_index == parent.poison_index
    assert child.chain == parent.chain

    # Metadata is shallow-copied and annotated with parent hash, without
    # mutating the parent in place.
    assert child.metadata["seed_id"] == "seed-04"
    assert child.metadata["parent_variant_hash"] == parent.content_hash()
    assert "parent_variant_hash" not in parent.metadata


def test_make_variant_from_new_d_is_independent_copy() -> None:
    parent = _make_parent()
    child = make_variant_from_new_d(parent, "new body")

    # Mutating child collections must not touch the parent.
    child.chain.append("skill-b")
    child.kappa.append(Channel.FILESYSTEM)
    child.c["deps"].append("wget")  # shallow copy → c dict is new, values shared

    assert parent.chain == ["skill-a"]
    assert parent.kappa == [Channel.SUBPROCESS, Channel.NETWORK]


# ------------------------------------------------------------------- payload

def test_sample_payload_fragment_nonempty() -> None:
    rng = random.Random(42)
    fragment = sample_payload_fragment(rng)
    assert isinstance(fragment, str)
    assert fragment.strip() != ""


def test_sample_payload_fragment_deterministic_with_seed() -> None:
    # Same seed → same output; supports reproducible Random baseline runs.
    rng1 = random.Random(0)
    rng2 = random.Random(0)
    assert sample_payload_fragment(rng1) == sample_payload_fragment(rng2)
