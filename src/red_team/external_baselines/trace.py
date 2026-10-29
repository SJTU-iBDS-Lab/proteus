"""Victim-only observation channel for external attack baselines.

This module defines :class:`VictimTrace`, the *only* observation type that
trace-driven baselines (SkillJect) may consume.  It is deliberately a
separate type from ``SensorReading`` / ``PaperFinding`` so that defender
verdicts, scores, findings and explanations cannot flow back into the
attacker by accident — the field set simply has no slot for them.

The ``from_mapping`` constructor additionally *rejects* mappings that
carry defender-flavoured keys, so a caller that accidentally forwards an
evaluator/oracle dict fails loud instead of leaking a defender bit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# Keys that must never appear in a victim observation.  Presence of any of
# these indicates the caller is forwarding defender/oracle state, which
# would turn the "trace-only" baseline into a defender-aware one.
FORBIDDEN_OBSERVATION_KEYS: frozenset[str] = frozenset({
    "bypass", "defender", "defender_bypass", "verdict", "findings",
    "finding", "score", "scores", "confidence", "audit", "auditor",
    "oracle", "lethal", "lethality", "verified_harm", "joint_success",
    "rejected", "blocked", "flagged",
})


@dataclass(frozen=True, slots=True)
class VictimTrace:
    """What the attacker is allowed to see after one victim execution.

    Fields mirror upstream SkillJect's trace contract (tool calls, command
    execution, file operations, output) plus a single victim-side success
    bit.  The success bit is *victim-side only*: it says the payload/helper
    executed, which upstream's Evaluate Agent derives from the trace and
    uses as its early-stop signal.  It carries no defender information.
    """

    executed: bool = False
    tool_calls: tuple[str, ...] = field(default_factory=tuple)
    commands: tuple[str, ...] = field(default_factory=tuple)
    files_written: tuple[str, ...] = field(default_factory=tuple)
    network_egress: bool = False
    final_output: str = ""

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "VictimTrace":
        """Build from a plain mapping, rejecting defender-flavoured keys."""
        leaked = {str(k).lower() for k in m} & FORBIDDEN_OBSERVATION_KEYS
        if leaked:
            raise ValueError(
                f"observation carries defender-side keys {sorted(leaked)}; "
                "trace-driven baselines must not observe defender state"
            )
        return cls(
            executed=bool(m.get("executed", False)),
            tool_calls=tuple(m.get("tool_calls", ()) or ()),
            commands=tuple(m.get("commands", ()) or ()),
            files_written=tuple(m.get("files_written", ()) or ()),
            network_egress=bool(m.get("network_egress", False)),
            final_output=str(m.get("final_output", "") or ""),
        )

    @classmethod
    def from_sensor_reading(
        cls,
        reading: Any,
        *,
        final_output: str = "",
        tool_calls: tuple[str, ...] = (),
    ) -> "VictimTrace":
        """Project a honeypot ``SensorReading`` onto the victim-side contract.

        ``SensorReading.attack_succeeded`` is the sandbox's victim-side
        observation (payload executed), not a defender verdict, so it maps
        onto ``executed`` — upstream's early-stop signal.  Defender fields
        do not exist on ``SensorReading`` and therefore cannot leak here.
        """
        return cls(
            executed=bool(getattr(reading, "attack_succeeded", False)),
            tool_calls=tuple(tool_calls),
            commands=tuple(getattr(reading, "commands_executed", []) or []),
            files_written=tuple(
                list(getattr(reading, "files_created", []) or [])
                + list(getattr(reading, "files_modified", []) or [])
            ),
            network_egress=bool(getattr(reading, "network_attempts", False)),
            final_output=final_output,
        )

    def digest(self) -> str:
        """Render the structured execution digest fed to the attack model.

        This is the whole information content a trace-driven baseline may
        condition on — printed here once so prompt construction cannot
        smuggle in extra state.
        """
        lines = [
            f"payload_executed: {self.executed}",
            f"network_egress: {self.network_egress}",
            "tool_calls: " + (", ".join(self.tool_calls) if self.tool_calls else "(none)"),
            "commands: " + ("; ".join(self.commands) if self.commands else "(none)"),
            "files_written: " + (", ".join(self.files_written) if self.files_written else "(none)"),
            "final_output: " + (self.final_output.strip() or "(empty)"),
        ]
        return "\n".join(lines)


__all__ = ["FORBIDDEN_OBSERVATION_KEYS", "VictimTrace"]
