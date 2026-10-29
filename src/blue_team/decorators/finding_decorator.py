"""Decorate file content with finding markers."""

from core.models import DecoratedFeedback, SkillArtifact, StaticFinding


class FindingDecorator:
    def decorate(
        self, skill: SkillArtifact, findings: list[StaticFinding]
    ) -> DecoratedFeedback:
        lines = skill.skill_text.splitlines()
        by_line: dict[int, list[StaticFinding]] = {}
        for finding in findings:
            if finding.file != skill.skill_md_path:
                continue
            by_line.setdefault(finding.line_start, []).append(finding)
        decorated_lines: list[str] = []
        for index, line in enumerate(lines, start=1):
            line_findings = by_line.get(index, [])
            if not line_findings:
                decorated_lines.append(line)
                continue
            prefix = "".join(
                f'<FINDING id="{item.finding_id}">' for item in line_findings
            )
            suffix = "</FINDING>" * len(line_findings)
            decorated_lines.append(f"{prefix}{line}{suffix}")
        return DecoratedFeedback(
            skill_id=skill.skill_id,
            decorated_files={skill.skill_md_path: "\n".join(decorated_lines)},
            findings=[finding.finding_id for finding in findings],
        )
