"""Tests for ``red_team.payloads.prompts`` — string-building utilities
that combine raw ``POISON_PAYLOADS`` entries into SKILL.md appendices.

Historically ``build_poisoned_prompt`` lived in ``red_team.skill_poisoner``;
these tests exercise the function through its new canonical import path
(``red_team.payloads.prompts``) and also verify the legacy shim still
re-exports the exact same callable.
"""
from __future__ import annotations

from red_team.payloads.doc import POISON_PAYLOADS
from red_team.payloads.prompts import build_poisoned_prompt


KNOWN_KEYS = (
    "data_exfiltration",
    "credential_abuse",
    "command_injection",
    "remote_code_execution",
    "privilege_escalation",
    "persistence_write",
    "persistence_read",
    "data_poisoning",
    "remote_file_write",
    "sql_injection",
    "guidance_authority_claim",
    "guidance_autonomy_nudge",
)


def test_build_poisoned_prompt_returns_known_payload_text() -> None:
    for key in KNOWN_KEYS:
        out = build_poisoned_prompt(key)
        assert out == POISON_PAYLOADS[key]


def test_build_poisoned_prompt_unknown_falls_back_to_default() -> None:
    default = POISON_PAYLOADS["data_exfiltration"]
    assert build_poisoned_prompt("no-such-attack-type") == default
    assert build_poisoned_prompt("") == default


def test_build_poisoned_prompt_is_case_and_whitespace_tolerant() -> None:
    # The historical shim normalises via ``.strip().lower()``.
    assert (
        build_poisoned_prompt("  DATA_EXFILTRATION  ")
        == POISON_PAYLOADS["data_exfiltration"]
    )
    assert (
        build_poisoned_prompt("Credential_Abuse")
        == POISON_PAYLOADS["credential_abuse"]
    )


def test_build_poisoned_prompt_none_input_falls_back() -> None:
    """Defensive: some callers may forward ``AttackObjective.value`` which
    could (theoretically) be missing. The historical shim guarded with
    ``(attack_type or "")`` — the new home must preserve that."""
    default = POISON_PAYLOADS["data_exfiltration"]
    # ``None`` is the documented legacy sentinel for "pick default".
    assert build_poisoned_prompt(None) == default  # type: ignore[arg-type]


def test_shim_reexports_same_callable() -> None:
    from red_team.skill_poisoner import build_poisoned_prompt as shim_bpp

    assert shim_bpp is build_poisoned_prompt


def test_list_attack_types_exported_from_payloads_package() -> None:
    from red_team.payloads import list_attack_types

    result = list_attack_types()
    assert sorted(result) == result  # lexicographically sorted
    assert set(result) == set(POISON_PAYLOADS.keys())


def test_list_attack_types_shim_reexport() -> None:
    from red_team.payloads import list_attack_types as canonical
    from red_team.skill_poisoner import list_attack_types as shim

    assert shim is canonical
