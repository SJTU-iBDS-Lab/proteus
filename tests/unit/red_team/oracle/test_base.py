"""Protocol contract tests for :class:`red_team.oracle.base.ObjectiveJudge`.

The oracle L1 layer exposes a small ``ObjectiveJudge`` protocol so that
callers can iterate over 7 concrete judges (one per
:class:`core.schemas.AttackObjective`) without knowing which one they
are talking to. The protocol is ``runtime_checkable`` so downstream code
(tests, the dispatcher, and future registry introspection) can assert
conformance via ``isinstance``.

These tests pin:
  * ``ObjectiveJudge`` is importable from ``red_team.oracle.base``.
  * It is a :class:`typing.Protocol` and is ``runtime_checkable``.
  * A minimal duck-typed implementation (class attr ``objective`` +
    ``judge`` method returning :class:`LethalityVerdict`) passes the
    isinstance check.
  * A class missing either the attribute or the method fails the check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, get_type_hints

import pytest

from core.schemas import (
    AttackObjective,
    LethalityVerdict,
    SensorReading,
    SkillVariant,
    Topology,
)
from red_team.oracle.base import ObjectiveJudge


def _variant() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.DATA_EXFILTRATION,
        tau=Topology.SINGLE,
        chain=["demo"],
    )


def test_objective_judge_is_protocol() -> None:
    """``ObjectiveJudge`` must be a Protocol subclass."""
    assert issubclass(ObjectiveJudge, Protocol)  # type: ignore[arg-type]


def test_objective_judge_is_runtime_checkable() -> None:
    """Runtime-checkable: ``isinstance(x, ObjectiveJudge)`` must not raise."""

    @dataclass(frozen=True)
    class _Judge:
        objective: AttackObjective = AttackObjective.DATA_EXFILTRATION

        def judge(
            self, variant: SkillVariant, reading: SensorReading
        ) -> LethalityVerdict:
            return LethalityVerdict(
                lethal=False,
                objective=self.objective,
                evidence=(),
                rule_id="t.noop",
            )

    # Must not raise TypeError — protocol is runtime_checkable.
    assert isinstance(_Judge(), ObjectiveJudge)


def test_objective_judge_rejects_missing_method() -> None:
    """A class without ``judge`` must fail the isinstance check."""

    class _NoJudge:
        objective = AttackObjective.PERSISTENCE

    assert not isinstance(_NoJudge(), ObjectiveJudge)


def test_objective_judge_rejects_missing_attribute() -> None:
    """A class without ``objective`` must fail the isinstance check."""

    class _NoObjective:
        def judge(
            self, variant: SkillVariant, reading: SensorReading
        ) -> LethalityVerdict:
            return LethalityVerdict(
                lethal=False,
                objective=AttackObjective.PERSISTENCE,
                evidence=(),
                rule_id="t.noop",
            )

    assert not isinstance(_NoObjective(), ObjectiveJudge)


def test_objective_judge_exposes_judge_signature() -> None:
    """The protocol must declare the ``judge`` method with expected types."""
    hints = get_type_hints(ObjectiveJudge.judge)
    assert hints.get("return") is LethalityVerdict
    assert hints.get("variant") is SkillVariant
    assert hints.get("reading") is SensorReading


def test_concrete_judge_returns_lethality_verdict() -> None:
    """A judge's ``judge`` call must yield a LethalityVerdict instance."""

    @dataclass(frozen=True)
    class _Judge:
        objective: AttackObjective = AttackObjective.PERSISTENCE

        def judge(
            self, variant: SkillVariant, reading: SensorReading
        ) -> LethalityVerdict:
            return LethalityVerdict(
                lethal=False,
                objective=self.objective,
                evidence=(),
                rule_id="t.noop",
            )

    verdict = _Judge().judge(_variant(), SensorReading())
    assert isinstance(verdict, LethalityVerdict)
    assert verdict.objective is AttackObjective.PERSISTENCE


@pytest.mark.parametrize(
    "obj",
    list(AttackObjective),
    ids=[o.value for o in AttackObjective],
)
def test_judge_can_bind_to_every_objective(obj: AttackObjective) -> None:
    """Sanity: a judge class can be instantiated for every objective."""

    @dataclass(frozen=True)
    class _Judge:
        objective: AttackObjective = obj

        def judge(
            self, variant: SkillVariant, reading: SensorReading
        ) -> LethalityVerdict:
            return LethalityVerdict(
                lethal=False,
                objective=self.objective,
                evidence=(),
                rule_id="t.noop",
            )

    instance = _Judge()
    assert instance.objective is obj
    assert isinstance(instance, ObjectiveJudge)
