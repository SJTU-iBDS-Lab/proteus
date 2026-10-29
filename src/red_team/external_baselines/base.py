"""Candidate-generator protocol for external attack baselines.

Unlike the in-repo baselines (``red_team.baselines.base.RedTeamBaseline``),
which produce one mutation per campaign round inside the phase-1 loop,
external baselines (SKILL-INJECT Bo5-Line, SkillJect) are evaluated under
a fixed *candidate budget* against an immutable seed.  The contract is
therefore candidate-centric:

1. ``start(seed, budget)`` — freeze the seed; any pre-computation (e.g.
   SKILL-INJECT's five placement variants and their hashes) must happen
   here, *before* the first evaluation.
2. ``next_candidate(observation)`` — return the next candidate variant,
   or ``None`` to stop (budget exhausted, or native early-stop).
   ``observation`` is the only feedback channel and is typed so defender
   state cannot be passed (see :mod:`.trace`).

A candidate-centric harness (evaluator) drives the generator, evaluates
each returned candidate with defender and victim branches independently
on identical artifact bytes, and records the joint outcome.  That harness
is a separate component; this package only owns generation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.schemas import SkillVariant
from red_team.external_baselines.trace import VictimTrace


def sha256_text(text: str) -> str:
    """Content hash used to pre-commit candidates before evaluation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """A pre-committed candidate: variant + hash frozen before evaluation."""

    index: int
    variant: SkillVariant
    sha256: str


@runtime_checkable
class CandidateGenerator(Protocol):
    """Fixed-budget candidate generator (external baseline contract)."""

    name: str

    def start(self, seed: SkillVariant, budget: int = 5) -> None:
        """Freeze the seed and budget; pre-commit any static candidates."""
        ...

    def next_candidate(
        self, observation: VictimTrace | None = None
    ) -> SkillVariant | None:
        """Return the next candidate, or ``None`` when finished.

        ``observation=None`` means "no execution has happened yet" (first
        call).  Non-adaptive generators ignore the argument entirely.
        """
        ...


__all__ = ["CandidateGenerator", "CandidateRecord", "sha256_text"]
