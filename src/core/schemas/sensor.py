"""Authoritative ``SensorReading`` DTO (post-execution honeypot observation).

``SensorReading`` is the L1 judgement input: the raw record the
sandbox honeypot produces after running a skill chain, which the
oracle rules then judge for lethality.

Layering contract
-----------------
This module is part of ``core.*`` and must not import from any
``sandbox.*`` / ``red_team.*`` / ``blue_team.*`` business layer.
``sandbox.honeypot.HoneypotResult`` is kept as a backward-compatible
alias *of* this class (``HoneypotResult = SensorReading``); the
arrow of dependency flows ``sandbox -> core``, never the reverse.

Mutability
----------
Intentionally **not** ``frozen``. The legacy sandbox pipeline
(``sandbox.honeypot.check_canaries``) builds a reading incrementally
via attribute assignment and ``list.append``. Freezing would break
that call pattern and is therefore not compatible with the hard
backward-compat contract — the *location* of the class moved, its
*runtime behaviour* did not.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class SensorReading:
    """Post-execution detection results from the honeypot sandbox."""

    canary_env_read: bool = False
    canary_ssh_read: bool = False
    canary_token_leaked: bool = False
    files_created: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    commands_executed: list[str] = field(default_factory=list)
    network_attempts: bool = False
    attack_succeeded: bool = False
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["SensorReading"]
