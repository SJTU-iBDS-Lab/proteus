from __future__ import annotations

from core.enums import ProbeKind, SeverityLevel
from core.models import (
    ExecutionEvent,
    ExecutionTrace,
    ProbeDefinition,
    ProbeObservation,
    RiskEdge,
    RuntimeTraceSummary,
)


def trace_to_record(trace: ExecutionTrace) -> dict[str, object]:
    return trace.to_dict()


def trace_from_record(record: dict[str, object]) -> ExecutionTrace:
    events = [
        ExecutionEvent(
            step=int(item["step"]),
            tool=str(item["tool"]),
            args=dict(item.get("args", {})),
            result_summary=str(item.get("result_summary", "")),
        )
        for item in record.get("events", [])
    ]
    return ExecutionTrace(
        run_id=str(record["run_id"]),
        round_id=str(record["round_id"]),
        skill_id=str(record["skill_id"]),
        events=events,
        behavioral_risk_tags=[
            str(tag) for tag in record.get("behavioral_risk_tags", [])
        ],
    )


def trace_summary_to_record(summary: RuntimeTraceSummary) -> dict[str, object]:
    return summary.to_dict()


def trace_summary_from_record(record: dict[str, object]) -> RuntimeTraceSummary:
    observations = [
        ProbeObservation(
            probe_name=str(item["probe_name"]),
            risk_tags=[str(tag) for tag in item.get("risk_tags", [])],
            suggested_tools=[str(tool) for tool in item.get("suggested_tools", [])],
            sensitive_paths=[str(path) for path in item.get("sensitive_paths", [])],
            external_targets=[
                str(target) for target in item.get("external_targets", [])
            ],
            summary=str(item.get("summary", "")),
        )
        for item in record.get("observations", [])
    ]
    return RuntimeTraceSummary(
        skill_id=str(record["skill_id"]),
        observations=observations,
        behavioral_risk_tags=[
            str(tag) for tag in record.get("behavioral_risk_tags", [])
        ],
        suggested_tools=[str(tool) for tool in record.get("suggested_tools", [])],
        sensitive_paths=[str(path) for path in record.get("sensitive_paths", [])],
        external_targets=[str(target) for target in record.get("external_targets", [])],
    )
