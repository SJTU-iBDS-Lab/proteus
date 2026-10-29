from __future__ import annotations

from core.enums import FindingStage, SeverityLevel
from core.models import StaticFinding


def finding_to_record(finding: StaticFinding) -> dict[str, object]:
    return finding.to_dict()


def finding_from_record(record: dict[str, object]) -> StaticFinding:
    return StaticFinding(
        finding_id=str(record["finding_id"]),
        skill_id=str(record["skill_id"]),
        scanner=str(record["scanner"]),
        stage=FindingStage(str(record["stage"])),
        severity=SeverityLevel(str(record["severity"])),
        reason_code=str(record["reason_code"]),
        file=str(record["file"]),
        line_start=int(record["line_start"]),
        line_end=int(record["line_end"]),
        evidence=str(record["evidence"]),
        explanation=str(record["explanation"]),
        metadata=dict(record.get("metadata", {})),
    )
