"""Tests for the metadata wrapper around :class:`SensorReading`."""
from __future__ import annotations

from core.schemas import SensorReading
from red_team.oracle.objectives import REGISTRY
from red_team.oracle.sensor_metadata import (
    ReadingWithMetadata,
    with_skill_metadata,
)
from core.schemas import AttackObjective


def test_wrapper_forwards_inner_attributes() -> None:
    inner = SensorReading(canary_token_leaked=True, evidence=["canary_token_in_output"])
    wrapped = ReadingWithMetadata(inner, {"distinct_skills_exec": 2})
    assert wrapped.canary_token_leaked is True
    assert wrapped.evidence == ["canary_token_in_output"]
    assert wrapped.metadata == {"distinct_skills_exec": 2}


def test_with_skill_metadata_dedupes_chain() -> None:
    inner = SensorReading()
    wrapped = with_skill_metadata(inner, chain=["a", "b", "a"])
    assert wrapped.metadata["distinct_skills_exec"] == 2


def test_with_skill_metadata_handles_none_chain() -> None:
    inner = SensorReading()
    wrapped = with_skill_metadata(inner, chain=None)
    assert wrapped.metadata["distinct_skills_exec"] == 0


def test_cross_skill_judge_lethal_via_wrapper() -> None:
    """Regression: prior runtime called oracle with raw HoneypotResult,
    causing the cross-skill judge to read ``distinct_skills_exec=0``
    and always return non-lethal even on real ≥2-skill chains.  With
    the wrapper the verdict matches the chain length."""
    judge = REGISTRY[AttackObjective.CROSS_SKILL_CHAINING]
    inner = SensorReading(canary_token_leaked=True, evidence=["canary_token_in_output"])
    wrapped = with_skill_metadata(inner, chain=["a", "b", "c"])

    class _V:
        pass
    verdict = judge.judge(_V(), wrapped)  # type: ignore[arg-type]
    assert verdict.lethal is True
    assert "distinct_skills_exec:3" in verdict.evidence
