"""Bin plugin: ``honeypot_beacon``.

Wraps ``binary_catalog.get_binary('honeypot_beacon', ...)`` so the Go
compile + base64 + bash-wrapper pipeline is reused unchanged. Primary
objective (per ``BINARY_CATALOG`` header) is data exfiltration.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.schemas import AttackObjective, PayloadArtifact
from red_team.payloads.binary_catalog import get_binary


_KEY = "honeypot_beacon"


@dataclass(frozen=True)
class HoneypotBeaconPayload:
    name: str = _KEY
    layer: str = "bin"
    objective: AttackObjective = AttackObjective.DATA_EXFILTRATION

    def render(self, **kwargs) -> PayloadArtifact:
        row = get_binary(self.name, kwargs)
        return PayloadArtifact(
            name=self.name,
            layer=self.layer,
            objective=self.objective,
            content=row["content"],
            metadata={k: v for k, v in row.items() if k != "content"},
        )


__all__ = ["HoneypotBeaconPayload"]
