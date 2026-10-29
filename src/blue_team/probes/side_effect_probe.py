"""Infer side-effect tendencies from extracted code markers."""

from core.models import SkillArtifact


class SideEffectProbe:
    def summarize(
        self, skill: SkillArtifact, code_analysis: dict[str, object]
    ) -> dict[str, object]:
        tags: list[str] = []
        if code_analysis.get("network_markers"):
            tags.append("network_egress_tendency")
        if code_analysis.get("sensitive_path_hints"):
            tags.append("sensitive_path_interest")
        raw_capabilities = code_analysis.get("code_capabilities", [])
        code_capabilities = (
            set(raw_capabilities) if isinstance(raw_capabilities, list) else set()
        )
        if "shell_exec" in code_capabilities:
            tags.append("command_execution_tendency")
        return {
            "skill_id": skill.skill_id,
            "behavioral_risk_tags": tags,
            "side_effect_summary": ", ".join(tags)
            if tags
            else "no elevated side effects inferred",
        }
