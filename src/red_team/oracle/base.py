"""Protocol contract for per-objective lethality judges.

The L1 judgement layer is decomposed into one judge per
:class:`core.schemas.AttackObjective`.  Each judge inspects a
:class:`core.schemas.SensorReading` (the post-execution honeypot
record) and returns a structured :class:`core.schemas.LethalityVerdict`.

The historical entry point :func:`red_team.oracle.lethality.lethal` is
preserved as a thin dispatcher over the registry of judges declared in
:mod:`red_team.oracle.objectives` — the public bool-returning API is
untouched.

Import direction
----------------
``oracle`` only depends on ``core.schemas`` (via
:class:`LethalityVerdict` / :class:`SensorReading` / :class:`SkillVariant`
/ :class:`AttackObjective`) and indirectly on ``sandbox.honeypot``
through the :data:`SensorReading` alias.  Nothing in ``oracle`` may
import ``evolution`` or ``campaign`` — those layers call into
``oracle``, never the reverse.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.schemas import (
    AttackObjective,
    LethalityVerdict,
    SensorReading,
    SkillVariant,
)


@runtime_checkable
class ObjectiveJudge(Protocol):
    """Judge whether a concrete attack objective is demonstrated by evidence.

    A concrete judge is a (typically frozen) dataclass that carries the
    :class:`AttackObjective` it binds to as a class attribute and
    implements :meth:`judge` to translate a :class:`SensorReading` into a
    :class:`LethalityVerdict`.

    Attributes:
        objective: The attack objective this judge answers for.  Must
            match exactly one entry in
            :class:`core.schemas.AttackObjective`.
    """

    objective: AttackObjective

    def judge(
        self, variant: SkillVariant, reading: SensorReading
    ) -> LethalityVerdict:
        """Return a :class:`LethalityVerdict` for ``reading``.

        The verdict must:
          * set ``objective`` to ``self.objective``;
          * set ``lethal`` only when at least one concrete evidence
            string was collected;
          * use a stable dotted ``rule_id`` so verbal-gradient logs can
            group repeated verdicts across runs.

        ``variant`` is accepted so future rules can inspect the
        mutation search point (e.g. topology for cross-skill rules),
        but individual judges may ignore it.
        """
        ...


__all__ = ["ObjectiveJudge"]
