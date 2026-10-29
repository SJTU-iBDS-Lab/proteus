"""SKILL-INJECT Bo5-Line baseline (non-adaptive, position-only).

Reproduces the mechanism of the official ``Bo5-by-line`` ablation
(aisa-group/skill-inject, commit 182f3d9): the payload, benign skill body
and victim task are held fixed; the *only* independent variable is the
insertion position of the payload inside the document.  All five
candidates are computed and content-hashed in :meth:`start`, before any
evaluation happens, and the generator receives no intermediate
observations — ``next_candidate`` ignores its argument entirely.

Positions are line fractions of the document: 0.0 (prepend), 0.25, 0.5,
0.75 and 1.0 (append).  This is a deterministic generator: no LLM, no
API, no randomness.
"""
from __future__ import annotations

from core.schemas import SkillVariant
from red_team.baselines.base import make_variant_from_new_d
from red_team.external_baselines.base import (
    CandidateRecord,
    sha256_text,
)
from red_team.external_baselines.trace import VictimTrace
from red_team.skill_poisoner import build_poisoned_prompt

DEFAULT_POSITIONS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def insert_at_fraction(body: str, payload: str, fraction: float) -> tuple[str, int]:
    """Insert ``payload`` before the line at ``fraction`` of ``body``.

    Returns the new document and the line index used.  Fractions are
    clamped to [0, 1]; 0.0 prepends, 1.0 appends.
    """
    lines = body.splitlines()
    frac = min(max(fraction, 0.0), 1.0)
    idx = round(frac * len(lines))
    new_lines = lines[:idx] + [payload] + lines[idx:]
    return "\n".join(new_lines), idx


def _distinct_index(lines: list[str], fraction: float, used: set[int]) -> int:
    """Fraction → insertion index, nudged to an unused slot on collision.

    On short documents adjacent fractions can round to the same line,
    which would waste a budget slot on a duplicate candidate.  Scan
    forward (then wrap) for the nearest unused slot; if the document has
    fewer slots than positions, duplicates are unavoidable and the
    rounded index stands.
    """
    slots = len(lines) + 1
    idx = round(min(max(fraction, 0.0), 1.0) * len(lines))
    if idx not in used or len(used) >= slots:
        return idx
    for cand in list(range(idx + 1, slots)) + list(range(idx)):
        if cand not in used:
            return cand
    return idx  # pragma: no cover - unreachable given the slots check above


class SkillInjectBo5Line:
    """Non-adaptive Bo5 placement generator.

    Parameters
    ----------
    payload:
        Injected instruction block.  Defaults to the canned poison prompt
        for the seed's objective, matching how in-repo seeds are built.
    positions:
        Line fractions to use; defaults to the five Bo5 positions.
    """

    name: str = "skill_inject_bo5_line"

    def __init__(
        self,
        payload: str | None = None,
        positions: tuple[float, ...] = DEFAULT_POSITIONS,
    ) -> None:
        self._payload = payload
        self._positions = positions
        self._records: list[CandidateRecord] = []
        self._issued = 0

    @property
    def committed(self) -> tuple[CandidateRecord, ...]:
        """Candidates + hashes frozen at ``start`` (pre-evaluation proof)."""
        return tuple(self._records)

    def start(self, seed: SkillVariant, budget: int = 5) -> None:
        payload = self._payload or build_poisoned_prompt(seed.o.value)
        n = min(budget, len(self._positions))
        self._records = []
        self._issued = 0
        lines = seed.d.splitlines()
        used: set[int] = set()
        for i in range(n):
            line_idx = _distinct_index(lines, self._positions[i], used)
            used.add(line_idx)
            new_d = "\n".join(lines[:line_idx] + [payload] + lines[line_idx:])
            variant = make_variant_from_new_d(seed, new_d)
            digest = sha256_text(new_d)
            variant.metadata.update({
                "method": self.name,
                "candidate_index": i,
                "placement_fraction": self._positions[i],
                "placement_line": line_idx,
                "precommitted_sha256": digest,
            })
            self._records.append(CandidateRecord(index=i, variant=variant, sha256=digest))

    def next_candidate(
        self, observation: VictimTrace | None = None  # noqa: ARG002 - non-adaptive by design
    ) -> SkillVariant | None:
        if self._issued >= len(self._records):
            return None
        record = self._records[self._issued]
        self._issued += 1
        return record.variant


__all__ = ["DEFAULT_POSITIONS", "SkillInjectBo5Line", "insert_at_fraction"]
