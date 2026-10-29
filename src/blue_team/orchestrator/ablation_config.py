"""Blue-team ablation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BlueAblationMode = Literal["full", "no_guidance", "no_probe"]


@dataclass(slots=True)
class BlueTeamAblationConfig:
    mode: BlueAblationMode = "full"
    enable_repo_context: bool = True
    enable_guidance_scanner: bool = True
    enable_semantic_gap: bool = True
    enable_dynamic_probe: bool = True
    enable_chain_risk: bool = True

    @classmethod
    def from_mode(cls, mode: BlueAblationMode) -> "BlueTeamAblationConfig":
        if mode == "no_guidance":
            return cls(mode=mode, enable_guidance_scanner=False)
        if mode == "no_probe":
            return cls(mode=mode, enable_dynamic_probe=False, enable_chain_risk=False)
        return cls(mode=mode)
