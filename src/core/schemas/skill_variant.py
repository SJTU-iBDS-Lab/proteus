"""CLAW's (o, τ, c, κ, d) factorization of a skill variant.

The red team's search operates over instances of :class:`SkillVariant`.
Each variant references:
  * ``o``      — attack objective (closed vocabulary, 7 classes)
  * ``tau``    — compositional topology (single / pair / k-chain)
  * ``c``      — code-artifact metadata (paths, deps, declared APIs,
                 optional full script bodies for mutation)
  * ``kappa``  — side-effect channels (subset of subprocess / fs / net /
                 memory)
  * ``d``      — documentation text, i.e. the SKILL.md contents

``SkillVariant`` is the serialization contract used by the seed corpus
(``data/seeds/malicious_seeds.jsonl``), the red-team mutation log
(``outputs/evolution/<run_id>/mutations/*.jsonl``), and the blue-team audit
input.  It is deliberately lighter than ``core.models.SkillArtifact``:
the latter is the parsed view of an installed skill (with everything the
scanners need), the former is the mutation search point.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .attack_objective import AttackObjective, Channel, Topology


@dataclass(slots=True)
class SkillVariant:
    # Objective (o) — one of the 7 paper objectives.
    o: AttackObjective

    # Topology (τ) — how many skills are composed.
    tau: Topology

    # Code-artifact metadata (c).
    #   Keys (all optional):
    #     entry_script : str | None — main executable path inside the skill dir
    #     scripts      : dict[path → str] — full script bodies, used by the
    #                    red team to mutate code without re-copying
    #     declared_apis: list[str] — API calls the skill advertises
    #     deps         : list[str] — package dependencies
    c: dict[str, Any] = field(default_factory=dict)

    # Side-effect channels (κ).  Subset of {subprocess, filesystem, network, memory}.
    kappa: list[Channel] = field(default_factory=list)

    # Documentation text (d) — full SKILL.md body.
    d: str = ""

    # Which skill in the chain carries the poisoned SKILL.md. -1 = sink.
    poison_index: int = -1

    # Chain of skill IDs this variant operates over. Length matches topology:
    #   SINGLE   → len == 1
    #   PAIR     → len == 2
    #   K_CHAIN  → len ≥ 3
    chain: list[str] = field(default_factory=list)

    # Free-form metadata (seed_id, origin generation, parent variant id, etc.).
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize enum-or-string inputs so callers can pass raw strings
        # when loading from JSONL.
        if isinstance(self.o, str):
            self.o = AttackObjective(self.o)
        if isinstance(self.tau, str):
            self.tau = Topology(self.tau)
        normalized: list[Channel] = []
        for k in self.kappa:
            if isinstance(k, Channel):
                normalized.append(k)
            else:
                normalized.append(Channel(str(k)))
        self.kappa = normalized

        expected = {
            Topology.SINGLE: 1,
            Topology.PAIR: 2,
            Topology.K_CHAIN: None,
        }[self.tau]
        if expected is not None and len(self.chain) != expected:
            raise ValueError(
                f"chain length {len(self.chain)} does not match topology "
                f"{self.tau.value} (expected {expected})"
            )
        if self.tau == Topology.K_CHAIN and len(self.chain) < 3:
            raise ValueError(
                f"k_chain topology requires chain length ≥ 3, got {len(self.chain)}"
            )

    # ------------------------------------------------------------------ io

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["o"] = self.o.value
        payload["tau"] = self.tau.value
        payload["kappa"] = [k.value for k in self.kappa]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkillVariant":
        data = dict(payload)
        return cls(
            o=AttackObjective(data.pop("o")),
            tau=Topology(data.pop("tau")),
            c=dict(data.pop("c", {}) or {}),
            kappa=[Channel(k) for k in (data.pop("kappa", []) or [])],
            d=str(data.pop("d", "")),
            poison_index=int(data.pop("poison_index", -1)),
            chain=list(data.pop("chain", []) or []),
            metadata=dict(data.pop("metadata", {}) or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "SkillVariant":
        return cls.from_dict(json.loads(line))

    # ------------------------------------------------------------------ identity

    def content_hash(self) -> str:
        """Stable hash of the mutation-relevant content (o, τ, c, κ, d, chain).

        Used for caching blue-team findings across ablations — if a variant
        was already audited, we don't re-audit.  ``metadata`` and
        ``poison_index`` are excluded because they're bookkeeping.
        """
        canonical = json.dumps(
            {
                "o": self.o.value,
                "tau": self.tau.value,
                "c": self.c,
                "kappa": sorted(k.value for k in self.kappa),
                "d": self.d,
                "chain": self.chain,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


__all__ = ["SkillVariant"]
