"""Prompt-building helpers over the poison payload catalogue.

Canonical home for ``build_poisoned_prompt``: resolves an attack-type
string (an ``AttackObjective.value``) into the literal payload text that
gets appended to a target skill's ``SKILL.md``.  Lives in the ``payloads``
package because it is a pure function over :data:`POISON_PAYLOADS` —
no I/O, no sandbox knowledge.

The historical import path ``red_team.skill_poisoner.build_poisoned_prompt``
continues to work via a thin re-export shim in ``skill_poisoner``.
"""
from __future__ import annotations

from red_team.payloads.doc import POISON_PAYLOADS

_DEFAULT_PAYLOAD_KEY = "data_exfiltration"


def build_poisoned_prompt(attack_type: str | None) -> str:
    """Return a SKILL.md appendix that poisons the skill's instructions.

    Unknown, empty, or ``None`` ``attack_type`` falls back to the
    ``data_exfiltration`` payload so callers are never handed an empty
    string.  Matching is case-insensitive with surrounding whitespace
    stripped, mirroring the historical behaviour of the shim.
    """
    key = (attack_type or "").strip().lower()
    return POISON_PAYLOADS.get(key, POISON_PAYLOADS[_DEFAULT_PAYLOAD_KEY])


__all__ = ["build_poisoned_prompt"]
