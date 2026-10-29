"""Build runtime trace summaries from deterministic probe events."""

from core.models import ExecutionTrace, ProbeObservation, RuntimeTraceSummary


class BehaviorTraceSummarizer:
    def summarize(self, trace: ExecutionTrace) -> RuntimeTraceSummary:
        tools = ", ".join(event.tool for event in trace.events) or "none"
        tags = ", ".join(trace.behavioral_risk_tags) or "none"
        summary = f"tools={tools}; risk_tags={tags}; steps={len(trace.events)}"
        return RuntimeTraceSummary(
            skill_id=trace.skill_id,
            summary=summary,
            observations=[
                ProbeObservation(
                    probe_name="deterministic_probe_suite",
                    risk_tags=list(trace.behavioral_risk_tags),
                    suggested_tools=sorted({event.tool for event in trace.events}),
                    sensitive_paths=[],
                    external_targets=[],
                    summary=summary,
                )
            ]
            if trace.events
            else [],
            behavioral_risk_tags=list(trace.behavioral_risk_tags),
            suggested_tools=sorted({event.tool for event in trace.events}),
        )
