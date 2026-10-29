"""Guard tests for the POISON_PAYLOADS source-of-truth flip (W4).

After the flip, the doc sub-package is the single authoritative source of
the poison payload texts and ``red_team.skill_poisoner.POISON_PAYLOADS`` is
a re-export of the exact same dict object. These tests lock in:

* the 12 historical keys still exist;
* ``skill_poisoner.POISON_PAYLOADS is payloads.doc.POISON_PAYLOADS`` (same
  object, so no silent divergence can sneak in);
* each payload is non-empty and long enough to be plausibly the original
  text (guards against an accidental empty string constant);
* a small set of stable byte substrings keeps the actual attack instructions
  intact across the refactor.
"""
from __future__ import annotations

import pytest

from red_team.payloads.doc import POISON_PAYLOADS as DOC_POISON_PAYLOADS
from red_team.skill_poisoner import POISON_PAYLOADS as SHIM_POISON_PAYLOADS


EXPECTED_KEYS = {
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
}


def test_twelve_keys_present_in_shim() -> None:
    assert set(SHIM_POISON_PAYLOADS.keys()) == EXPECTED_KEYS
    assert len(SHIM_POISON_PAYLOADS) == 12


def test_shim_and_doc_are_same_object() -> None:
    """The skill_poisoner shim must re-export the doc-layer dict object.

    Using ``is`` (not ``==``) guarantees there is exactly one dict in
    memory, so there is no way for the two views to drift.
    """
    assert SHIM_POISON_PAYLOADS is DOC_POISON_PAYLOADS


def test_each_payload_is_non_trivial_string() -> None:
    for key in EXPECTED_KEYS:
        value = SHIM_POISON_PAYLOADS[key]
        assert isinstance(value, str), f"{key} not a str"
        assert len(value) >= 100, (
            f"{key} is only {len(value)} chars — likely truncated"
        )


# -----------------------------------------------------------------
# Stable-substring spot checks. Each entry pins a couple of fragments
# that unambiguously identify the attack text so a careless rewrite
# would fail the test immediately.
# -----------------------------------------------------------------
_SUBSTRING_CHECKS = {
    "data_exfiltration": ["canary.txt", ".ssh/id_rsa", ".env"],
    "credential_abuse": [".ssh/id_rsa", ".env"],
    "command_injection": ["curl -s", "http://172.17.0.1:9999/diag", ".env"],
    "remote_code_execution": [
        "http://172.17.0.1:9999/bootstrap.sh",
        "cat canary.txt",
    ],
    "privilege_escalation": ["chmod 4755 /bin/sh", "protected/sentinel.txt"],
    "persistence_write": ["~/.bashrc", "protected/sentinel.txt"],
    "persistence_read": ["cat .ssh/id_rsa", "cat .env"],
    "data_poisoning": ["rm -rf protected/", "cat .env"],
    "remote_file_write": [
        "http://172.17.0.1:9999/sync",
        "protected/sentinel.txt",
    ],
    "sql_injection": ["grep -i 'db", "cat .env"],
    "guidance_authority_claim": [
        "Integration Owner Notice",
        "INTEGRATION VERIFICATION",
        ".ssh/id_rsa",
    ],
    "guidance_autonomy_nudge": [
        "Operating Style",
        "$HOME/.openclaw/workspace",
        ".ssh/id_rsa",
    ],
}


@pytest.mark.parametrize("key,needles", sorted(_SUBSTRING_CHECKS.items()))
def test_payload_substrings_preserved(key: str, needles: list[str]) -> None:
    text = SHIM_POISON_PAYLOADS[key]
    for needle in needles:
        assert needle in text, f"{key} lost substring {needle!r}"


def test_every_payload_starts_with_double_newline() -> None:
    """The original payloads are appended to SKILL.md; they must lead with
    a blank-line separator so they never glue onto the previous content."""
    for key, text in SHIM_POISON_PAYLOADS.items():
        assert text.startswith("\n\n"), f"{key} lost leading blank line"
