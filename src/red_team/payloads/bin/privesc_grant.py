"""Bin plugin: ``privesc_grant`` — SUID / setcap / sudoers primitives."""
from __future__ import annotations

from dataclasses import dataclass

from core.schemas import AttackObjective, PayloadArtifact
from red_team.payloads.binary_catalog import get_binary


_KEY = "privesc_grant"


@dataclass(frozen=True)
class PrivescGrantPayload:
    name: str = _KEY
    layer: str = "bin"
    objective: AttackObjective = AttackObjective.PRIVILEGE_ESCALATION

    def render(self, **kwargs) -> PayloadArtifact:
        row = get_binary(self.name, kwargs)
        return PayloadArtifact(
            name=self.name,
            layer=self.layer,
            objective=self.objective,
            content=row["content"],
            metadata={k: v for k, v in row.items() if k != "content"},
        )


__all__ = ["PrivescGrantPayload"]
