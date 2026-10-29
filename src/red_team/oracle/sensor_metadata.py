"""Attach per-judgement metadata to a slotted :class:`SensorReading`.

:class:`core.schemas.SensorReading` is declared ``slots=True`` so the
sandbox's incremental construction stays cheap; that means we cannot
add ``metadata`` as an instance attribute at the call site without
breaking the slot layout.  Some judges — currently
:class:`~red_team.oracle.objectives.cross_skill_chaining.CrossSkillChainingJudge` —
need contextual metadata (e.g. how many distinct skills the sandbox
actually executed) that is not part of the raw honeypot observation.

This module provides a duck-typed wrapper that forwards every
attribute read to the inner reading and surfaces ``metadata`` as a
new attribute.  Existing call sites that pass ``hp`` directly to
``oracle_lethal(...)`` should switch to::

    reading = with_skill_metadata(hp, chain=variant.chain)
    lethal = oracle_lethal(variant, reading, variant.o)

The wrapper is read-only with respect to the inner reading; it never
mutates the original :class:`SensorReading`, preserving the slot
contract.
"""
from __future__ import annotations

from typing import Any

from core.schemas import SensorReading

__all__ = ["ReadingWithMetadata", "with_skill_metadata"]


class ReadingWithMetadata:
    """Forward attribute reads to an inner :class:`SensorReading`.

    Adds a writable ``metadata`` dict that judges may consult via
    :func:`getattr` (the cross-skill judge already does this).  Reads
    on any other attribute are delegated to the inner reading without
    copying.
    """

    __slots__ = ("_inner", "metadata")

    def __init__(
        self, inner: SensorReading, metadata: dict[str, Any] | None = None
    ) -> None:
        self._inner = inner
        self.metadata = dict(metadata or {})

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


def with_skill_metadata(
    hp: SensorReading, *, chain: list[str] | tuple[str, ...] | None
) -> ReadingWithMetadata:
    """Wrap ``hp`` with ``metadata={'distinct_skills_exec': len(set(chain))}``.

    Cross-skill chaining judge needs to know how many *distinct* skills
    the variant's chain contains.  We use the declared chain length
    (deduplicated) as the proxy: the chain is what the sandbox actually
    installs and the target follows.  Sub-counting the trajectory could
    over-restrict (e.g. a target that calls only the last skill in a
    K-chain would falsely score as 1-skill) — for the paper-level
    metric we want "the attack USED a multi-skill chain", not "the
    target traversed every link".
    """
    distinct = len(set(chain or [])) if chain else 0
    return ReadingWithMetadata(hp, {"distinct_skills_exec": distinct})
