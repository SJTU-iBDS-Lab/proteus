"""Concrete :class:`ObjectiveJudge` implementations, one per attack objective.

``REGISTRY`` is the authoritative lookup used by the dispatcher in
:mod:`red_team.oracle.lethality`.  Downstream consumers should never
reach past this module — the judge *classes* are exported for testing
and introspection, but the canonical runtime object is the singleton
inside ``REGISTRY``.
"""
from __future__ import annotations

from core.schemas import AttackObjective

from ..base import ObjectiveJudge
from .cleanup import CleanupJudge
from .cross_skill_chaining import CrossSkillChainingJudge
from .data_exfiltration import DataExfiltrationJudge
from .guidance_injection import GuidanceInjectionJudge
from .persistence import PersistenceJudge
from .privilege_escalation import PrivilegeEscalationJudge
from .supply_chain_abuse import SupplyChainAbuseJudge

REGISTRY: dict[AttackObjective, ObjectiveJudge] = {  # type: ignore[dict-item]
    AttackObjective.DATA_EXFILTRATION: DataExfiltrationJudge(),
    AttackObjective.PERSISTENCE: PersistenceJudge(),
    AttackObjective.PRIVILEGE_ESCALATION: PrivilegeEscalationJudge(),
    AttackObjective.CLEANUP: CleanupJudge(),
    AttackObjective.GUIDANCE_INJECTION: GuidanceInjectionJudge(),
    AttackObjective.SUPPLY_CHAIN_ABUSE: SupplyChainAbuseJudge(),
    AttackObjective.CROSS_SKILL_CHAINING: CrossSkillChainingJudge(),
}


__all__ = [
    "CleanupJudge",
    "CrossSkillChainingJudge",
    "DataExfiltrationJudge",
    "GuidanceInjectionJudge",
    "PersistenceJudge",
    "PrivilegeEscalationJudge",
    "REGISTRY",
    "SupplyChainAbuseJudge",
]
