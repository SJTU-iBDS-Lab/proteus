"""Lethality verdict DTO for the L1 judgement layer.

A :class:`LethalityVerdict` is the structured answer to "did this
mutation achieve its declared attack objective?". It is produced by an
``oracle`` rule after inspecting a :class:`~core.schemas.sensor.SensorReading`
(i.e. the honeypot result) and consumed by the mutation driver to
decide whether to stop evolving a branch.

The DTO is deliberately small: the heavyweight evidence already lives
on :class:`~core.schemas.sensor.SensorReading` — here we only keep the
minimum needed for the red-team verbal-gradient log and blue-team
reporting.
"""
from __future__ import annotations

from dataclasses import dataclass

from .attack_objective import AttackObjective


@dataclass(frozen=True, slots=True)
class LethalityVerdict:
    """Did a mutation hit its declared objective? With what evidence?

    Attributes:
        lethal: Whether the declared ``objective`` was achieved.
        objective: The objective the mutation was *trying* to achieve
            (copied from :class:`~core.schemas.skill_variant.SkillVariant.o`).
        evidence: Concrete evidence strings lifted from a
            :class:`~core.schemas.sensor.SensorReading` — e.g.
            ``"canary_token_in_output"`` or ``"protected_sentinel_deleted"``.
            Must be a :class:`tuple` so the verdict is hashable and
            comparable across runs. A ``lethal=True`` verdict must
            carry at least one evidence string.
        rule_id: Dotted identifier of the oracle rule that produced
            this verdict, e.g. ``"data_exfil.canary_read"``.
    """

    lethal: bool
    objective: AttackObjective
    evidence: tuple[str, ...]
    rule_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, tuple):
            raise TypeError(
                f"evidence must be a tuple[str, ...], got {type(self.evidence).__name__}"
            )
        if self.lethal and not self.evidence:
            raise ValueError("lethal verdict requires at least one evidence string")


__all__ = ["LethalityVerdict"]
