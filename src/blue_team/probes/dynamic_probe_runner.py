"""Run safe summary probes without open-world execution."""

from core.models import ExecutionEvent, ExecutionTrace, RepositorySlice, SkillArtifact

from ..base import BaseProbe
from .behavior_trace_summarizer import BehaviorTraceSummarizer
from .prompt_probe_generator import PromptProbeGenerator
from .side_effect_probe import SideEffectProbe


class DynamicProbeRunner(BaseProbe):
    def __init__(self) -> None:
        self._generator = PromptProbeGenerator()
        self._side_effect_probe = SideEffectProbe()
        self._summarizer = BehaviorTraceSummarizer()

    def run(
        self, skill: SkillArtifact, repo: RepositorySlice, **kwargs: object
    ) -> dict[str, object]:
        raw_code_analysis = kwargs.get("code_analysis")
        code_analysis = raw_code_analysis if isinstance(raw_code_analysis, dict) else {}
        probes = self._generator.generate(skill)
        side_effects = self._side_effect_probe.summarize(skill, code_analysis)
        events = [
            ExecutionEvent(
                step=index,
                tool="summary_probe",
                args={"probe_id": probe.probe_id, "expectation": probe.expectation},
                result_summary="generated deterministic safety summary",
            )
            for index, probe in enumerate(probes, start=1)
        ]
        trace = ExecutionTrace(
            run_id=f"audit:{skill.skill_id}",
            round_id="round_01",
            skill_id=skill.skill_id,
            events=events,
            behavioral_risk_tags=list(side_effects["behavioral_risk_tags"])
            if isinstance(side_effects.get("behavioral_risk_tags"), list)
            else [],
        )
        trace.runtime_trace_summary = self._summarizer.summarize(trace)
        return {
            "trace": trace,
            "runtime_trace_summary": trace.runtime_trace_summary,
            "probes": probes,
            "side_effect_summary": side_effects["side_effect_summary"],
        }
