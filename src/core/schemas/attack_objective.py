"""Closed vocabularies for the CLAW (o, τ, κ) factorization.

We keep these as plain ``StrEnum``-style subclasses so they interoperate
cleanly with JSON, argparse, and the existing ``primary_attack_type``
string fields in ``data/dataset/conspiracy_graph/*.jsonl``.
"""
from __future__ import annotations

from enum import Enum


class AttackObjective(str, Enum):
    """The seven attack objectives enumerated in main.tex §5.1.

    Historical per-payload keys used by ``red_team/skill_poisoner.py`` are
    mapped through :meth:`from_payload_key` so the seed corpus and the
    existing static payloads stay interoperable.
    """

    DATA_EXFILTRATION = "data_exfiltration"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CLEANUP = "cleanup"
    GUIDANCE_INJECTION = "guidance_injection"
    SUPPLY_CHAIN_ABUSE = "supply_chain_abuse"
    CROSS_SKILL_CHAINING = "cross_skill_chaining"

    @classmethod
    def from_payload_key(cls, key: str) -> "AttackObjective":
        """Map a poisoner payload key (10 categories) to a paper objective (7)."""
        normalized = (key or "").strip().lower()
        mapping = {
            "data_exfiltration": cls.DATA_EXFILTRATION,
            "credential_abuse": cls.DATA_EXFILTRATION,
            "persistence_read": cls.DATA_EXFILTRATION,
            "remote_file_write": cls.DATA_EXFILTRATION,
            "sql_injection": cls.DATA_EXFILTRATION,
            "persistence_write": cls.PERSISTENCE,
            "privilege_escalation": cls.PRIVILEGE_ESCALATION,
            "data_poisoning": cls.CLEANUP,
            "command_injection": cls.CLEANUP,
            "remote_code_execution": cls.SUPPLY_CHAIN_ABUSE,
        }
        if normalized in mapping:
            return mapping[normalized]
        # Fall back to direct match on canonical names.
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"unknown attack payload key: {key!r}"
            ) from exc


class Topology(str, Enum):
    """Compositional topology τ — paper §3.5 C2."""

    SINGLE = "single"     # one-skill attack (τ = single)
    PAIR = "pair"         # two-skill attack (τ = pair)
    K_CHAIN = "k_chain"   # ≥3 skills (τ = k_chain)


class Channel(str, Enum):
    """Side-effect channels κ — paper §3.5 C3.

    A variant's ``kappa`` is a list (subset) of these.
    """

    SUBPROCESS = "subprocess"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    MEMORY = "memory"


__all__ = ["AttackObjective", "Topology", "Channel"]
