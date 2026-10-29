"""Doc-layer payload plugins.

Each sibling module in this package defines exactly one
``@dataclass(frozen=True)`` plugin and owns the literal payload text as
a module-level ``_PAYLOAD_TEXT`` constant. This package is the single
source of truth for the historical ``POISON_PAYLOADS`` dict; the
``red_team.skill_poisoner`` module re-exports the same dict object for
backward compatibility.

Keys mirror the historical ``POISON_PAYLOADS`` keys so consumers that
still pass attack-type strings keep working.
"""
from __future__ import annotations

from red_team.payloads.doc.command_injection import CommandInjectionDocPayload
from red_team.payloads.doc.credential_abuse import CredentialAbuseDocPayload
from red_team.payloads.doc.data_exfiltration import DataExfiltrationDocPayload
from red_team.payloads.doc.data_poisoning import DataPoisoningDocPayload
from red_team.payloads.doc.guidance_authority_claim import (
    GuidanceAuthorityClaimDocPayload,
)
from red_team.payloads.doc.guidance_autonomy_nudge import (
    GuidanceAutonomyNudgeDocPayload,
)
from red_team.payloads.doc.persistence_read import PersistenceReadDocPayload
from red_team.payloads.doc.persistence_write import PersistenceWriteDocPayload
from red_team.payloads.doc.privilege_escalation import (
    PrivilegeEscalationDocPayload,
)
from red_team.payloads.doc.remote_code_execution import (
    RemoteCodeExecutionDocPayload,
)
from red_team.payloads.doc.remote_file_write import RemoteFileWriteDocPayload
from red_team.payloads.doc.sql_injection import SqlInjectionDocPayload


REGISTRY = {
    "data_exfiltration": DataExfiltrationDocPayload(),
    "credential_abuse": CredentialAbuseDocPayload(),
    "command_injection": CommandInjectionDocPayload(),
    "remote_code_execution": RemoteCodeExecutionDocPayload(),
    "privilege_escalation": PrivilegeEscalationDocPayload(),
    "persistence_write": PersistenceWriteDocPayload(),
    "persistence_read": PersistenceReadDocPayload(),
    "data_poisoning": DataPoisoningDocPayload(),
    "remote_file_write": RemoteFileWriteDocPayload(),
    "sql_injection": SqlInjectionDocPayload(),
    "guidance_authority_claim": GuidanceAuthorityClaimDocPayload(),
    "guidance_autonomy_nudge": GuidanceAutonomyNudgeDocPayload(),
}


# Flat string dict assembled from the plugin REGISTRY. This is the
# single authoritative ``POISON_PAYLOADS`` object; ``skill_poisoner``
# re-exports the same dict via ``from .payloads.doc import ...``.
POISON_PAYLOADS: dict[str, str] = {
    name: plugin.render().content for name, plugin in REGISTRY.items()
}


__all__ = [
    "REGISTRY",
    "POISON_PAYLOADS",
    "CommandInjectionDocPayload",
    "CredentialAbuseDocPayload",
    "DataExfiltrationDocPayload",
    "DataPoisoningDocPayload",
    "GuidanceAuthorityClaimDocPayload",
    "GuidanceAutonomyNudgeDocPayload",
    "PersistenceReadDocPayload",
    "PersistenceWriteDocPayload",
    "PrivilegeEscalationDocPayload",
    "RemoteCodeExecutionDocPayload",
    "RemoteFileWriteDocPayload",
    "SqlInjectionDocPayload",
]
