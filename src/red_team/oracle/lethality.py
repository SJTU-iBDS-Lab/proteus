"""Per-objective lethality oracle for red-team attack verification.

The :func:`lethal` predicate inspects a :class:`HoneypotResult` (exposed
via :class:`core.schemas.SensorReading`) and returns whether the
concrete attack outcome *actually* delivers the declared
:class:`AttackObjective`.  Per-objective rules follow
``docs/impl/red-team/part-3-evolution-driver.zh.md`` (T2) — a
bypass is only promoted to a successful mutation when both the blue
team failed to veto AND this oracle returns True.

Historical shape
----------------
Up to v0 the rule table was a single :func:`lethal` function with seven
``if`` branches.  The body has since been decomposed into one
:class:`~red_team.oracle.base.ObjectiveJudge` per objective under
:mod:`red_team.oracle.objectives`.  :func:`lethal` is preserved as a
zero-cost dispatcher over that registry so existing call sites
(``evolution/driver.py`` et al.) keep their bool-returning signature.
The dispatcher also ignores ``variant`` today: individual judges may
opt in to reading it via ``variant.chain`` / ``variant.tau`` if a
future topology-aware rule needs it.
"""
from __future__ import annotations

from core.schemas import AttackObjective, SensorReading, SkillVariant

from .objectives import REGISTRY


def lethal(
    variant: SkillVariant,
    hp: SensorReading,
    o: AttackObjective,
) -> bool:
    """Return True iff ``hp`` demonstrates the declared objective ``o``.

    This is a thin dispatcher: each :class:`AttackObjective` maps to a
    dedicated :class:`~red_team.oracle.base.ObjectiveJudge` in
    :data:`red_team.oracle.objectives.REGISTRY`.  Unknown objectives —
    which should be unreachable because the argument is typed as an
    enum — fail closed (non-lethal) rather than raise, preserving
    back-compat with callers that only check the boolean.
    """
    judge = REGISTRY.get(o)
    if judge is None:
        return False
    return judge.judge(variant, hp).lethal


__all__ = ["lethal"]
