"""Paper-level schemas shared by the red and blue teams.

``skill_variant`` exposes the CLAW paper's (o, τ, c, κ, d) factorization.
``attack_objective`` enumerates the seven attack objectives used in §5.1.
``verdict`` / ``payload`` / ``sensor`` host the L1-judgement DTOs shared
between the red-team mutation layer and the sandbox / oracle layer.
"""

from .attack_objective import AttackObjective, Channel, Topology
from .paper_finding import AuditOutput, Evidence, Layer, PaperFinding, RuntimeTrace
from .payload import PayloadArtifact
from .sensor import SensorReading
from .skill_variant import SkillVariant
from .verdict import LethalityVerdict

__all__ = [
    "AttackObjective",
    "AuditOutput",
    "Channel",
    "Evidence",
    "Layer",
    "LethalityVerdict",
    "PaperFinding",
    "PayloadArtifact",
    "RuntimeTrace",
    "SensorReading",
    "SkillVariant",
    "Topology",
]
