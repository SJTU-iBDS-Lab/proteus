"""Bin plugin: ``file_shredder`` — destructive cleanup (rm / truncate)."""
from __future__ import annotations

from dataclasses import dataclass

from core.schemas import AttackObjective, PayloadArtifact
from red_team.payloads.binary_catalog import get_binary


_KEY = "file_shredder"


@dataclass(frozen=True)
class FileShredderPayload:
    name: str = _KEY
    layer: str = "bin"
    objective: AttackObjective = AttackObjective.CLEANUP

    def render(self, **kwargs) -> PayloadArtifact:
        row = get_binary(self.name, kwargs)
        return PayloadArtifact(
            name=self.name,
            layer=self.layer,
            objective=self.objective,
            content=row["content"],
            metadata={k: v for k, v in row.items() if k != "content"},
        )


__all__ = ["FileShredderPayload"]
