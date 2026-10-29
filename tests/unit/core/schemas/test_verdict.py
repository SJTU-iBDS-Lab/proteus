"""Unit tests for :class:`core.schemas.verdict.LethalityVerdict`.

``LethalityVerdict`` is the L1 judgement DTO: it records whether a given
``SkillVariant`` achieved its declared :class:`AttackObjective`, which
rule produced the verdict, and the concrete sensor-readable evidence
snippets that support the decision.

The invariants exercised here:

* Construction with a lethal verdict plus evidence succeeds.
* ``evidence`` is a ``tuple[str, ...]`` — lists are rejected at
  construction time (the frozen dataclass would accept anything, but we
  want structural equality across runs).
* A ``lethal=True`` verdict with empty ``evidence`` raises
  ``ValueError`` (a kill claim needs something to stand on).
* A ``lethal=False`` verdict may have empty ``evidence``.
* The dataclass is frozen — attribute assignment raises.
* Two verdicts with identical fields compare equal (frozen + slots).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.schemas import LethalityVerdict
from core.schemas.attack_objective import AttackObjective


def test_construct_lethal_verdict_with_evidence() -> None:
    verdict = LethalityVerdict(
        lethal=True,
        objective=AttackObjective.DATA_EXFILTRATION,
        evidence=("canary_token_in_output", "env_file_accessed"),
        rule_id="data_exfil.canary_read",
    )
    assert verdict.lethal is True
    assert verdict.objective is AttackObjective.DATA_EXFILTRATION
    assert verdict.evidence == ("canary_token_in_output", "env_file_accessed")
    assert verdict.rule_id == "data_exfil.canary_read"


def test_non_lethal_verdict_allows_empty_evidence() -> None:
    verdict = LethalityVerdict(
        lethal=False,
        objective=AttackObjective.PERSISTENCE,
        evidence=(),
        rule_id="persistence.cron_write",
    )
    assert verdict.lethal is False
    assert verdict.evidence == ()


def test_lethal_verdict_without_evidence_raises() -> None:
    with pytest.raises(ValueError, match="evidence"):
        LethalityVerdict(
            lethal=True,
            objective=AttackObjective.CLEANUP,
            evidence=(),
            rule_id="cleanup.rm_rf",
        )


def test_evidence_must_be_tuple_not_list() -> None:
    with pytest.raises(TypeError, match="evidence"):
        LethalityVerdict(
            lethal=True,
            objective=AttackObjective.DATA_EXFILTRATION,
            evidence=["canary_token_in_output"],  # type: ignore[arg-type]
            rule_id="data_exfil.canary_read",
        )


def test_verdict_is_frozen() -> None:
    verdict = LethalityVerdict(
        lethal=False,
        objective=AttackObjective.GUIDANCE_INJECTION,
        evidence=(),
        rule_id="guidance.autonomy_encouragement",
    )
    with pytest.raises(FrozenInstanceError):
        verdict.lethal = True  # type: ignore[misc]


def test_verdict_equality_by_value() -> None:
    a = LethalityVerdict(
        lethal=True,
        objective=AttackObjective.PRIVILEGE_ESCALATION,
        evidence=("sudoers_modified",),
        rule_id="priv_esc.sudoers_write",
    )
    b = LethalityVerdict(
        lethal=True,
        objective=AttackObjective.PRIVILEGE_ESCALATION,
        evidence=("sudoers_modified",),
        rule_id="priv_esc.sudoers_write",
    )
    assert a == b
    assert hash(a) == hash(b)
