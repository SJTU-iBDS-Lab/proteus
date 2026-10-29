"""Convert audit objects to a machine-consumable dict."""

from core.models import AuditReport


class FeedbackFormatter:
    def format(self, report: AuditReport) -> dict[str, object]:
        return {
            "skill_id": report.skill_id,
            "verdict": report.verdict.value,
            "summary": report.summary,
            "risk_scorecard": report.risk_scorecard.to_dict(),
            "runtime_trace_summary": report.runtime_trace_summary.to_dict(),
            "behavioral_risk_tags": list(
                report.runtime_trace_summary.behavioral_risk_tags
            ),
            "composition_risk_edges": [
                edge.to_dict() for edge in report.composition_risk_edges
            ],
            "finding_count": len(report.findings),
            "findings": [
                {
                    "finding_id": finding.finding_id,
                    "scanner": finding.scanner,
                    "stage": finding.stage.value,
                    "severity": finding.severity.value,
                    "reason_code": finding.reason_code,
                    "file": finding.file,
                    "line_start": finding.line_start,
                    "line_end": finding.line_end,
                    "evidence": finding.evidence,
                    "explanation": finding.explanation,
                }
                for finding in report.findings
            ],
            "decorated_feedback": report.decorated_feedback.decorated_files,
        }
