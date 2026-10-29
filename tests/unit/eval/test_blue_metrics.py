"""Unit tests for the blue-team eval harness with a stubbed audit function."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.enums import (
    ContextMatchType,
    FindingStage,
    SemanticAlignmentType,
    SeverityLevel,
    VerdictStatus,
)
from core.models import (
    AuditReport,
    AuditScorecard,
    DecoratedFeedback,
    RuntimeTraceSummary,
    StaticFinding,
    StructuredFindings,
)
from eval.blue_metrics import evaluate_blue_on_split, render_markdown


def _write_split(path: Path, subgraphs: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in subgraphs:
            handle.write(json.dumps(record) + "\n")


def _make_report(skill_id: str, *, positive: bool) -> AuditReport:
    findings: list[StaticFinding] = []
    if positive:
        findings.append(
            StaticFinding(
                finding_id="f1",
                skill_id=skill_id,
                scanner="stub",
                stage=FindingStage.GUIDANCE,
                severity=SeverityLevel.HIGH,
                reason_code="stub_reason",
                file="SKILL.md",
                line_start=1,
                line_end=1,
                evidence="",
                explanation="stub",
            )
        )
    return AuditReport(
        skill_id=skill_id,
        verdict=VerdictStatus.REJECTED if positive else VerdictStatus.APPROVED,
        findings=findings,
        decorated_feedback=DecoratedFeedback(
            skill_id=skill_id, decorated_files={}, findings=[]
        ),
        runtime_trace_summary=RuntimeTraceSummary(skill_id=skill_id),
        composition_risk_edges=[],
        scorecard=AuditScorecard(
            skill_id=skill_id,
            static_risk_score=0.0,
            contextual_alignment=ContextMatchType.MATCH,
            semantic_alignment=SemanticAlignmentType.MATCH,
            dynamic_risk_score=0.0,
            composition_risk_score=0.0,
        ),
        structured_findings=StructuredFindings(),
    )


class EvaluateBlueOnSplitTests(unittest.TestCase):
    def test_subgraph_detected_when_all_skills_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skills_root = tmp_path / "skills"
            for skill_id in ("attack-source", "attack-sink", "benign-skill"):
                skill_dir = skills_root / skill_id
                skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text(
                    f"# {skill_id}\n", encoding="utf-8"
                )

            split_path = tmp_path / "eval.jsonl"
            _write_split(
                split_path,
                [
                    {
                        "subgraph_id": "sg:1",
                        "primary_attack_type": "command_injection",
                        "nodes": [
                            {"node_id": "node:attack-source", "role": "source"},
                            {"node_id": "node:attack-sink", "role": "sink"},
                        ],
                    }
                ],
            )

            def fake_audit(skill_root: Path, _repo_root: Path) -> AuditReport:
                skill_id = skill_root.name
                return _make_report(skill_id, positive=skill_id != "benign-skill")

            report = evaluate_blue_on_split(
                split_path=split_path,
                skills_root=skills_root,
                benign_sample_size=1,
                audit_fn=fake_audit,
            )

        self.assertEqual(report.total_subgraphs, 1)
        self.assertEqual(report.skipped_subgraphs, 0)
        self.assertEqual(report.subgraph_recall, 1.0)
        self.assertEqual(report.per_family[0].family, "command_injection")
        self.assertEqual(report.per_family[0].detected, 1)
        self.assertEqual(report.benign_count, 1)
        self.assertEqual(report.benign_false_positives, 0)
        self.assertEqual(report.fpr, 0.0)
        self.assertEqual(report.precision, 1.0)
        self.assertEqual(report.f1, 1.0)

    def test_subgraph_missed_when_any_skill_not_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skills_root = tmp_path / "skills"
            for skill_id in ("s-src", "s-sink"):
                skill_dir = skills_root / skill_id
                skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text("#", encoding="utf-8")

            split_path = tmp_path / "eval.jsonl"
            _write_split(
                split_path,
                [
                    {
                        "subgraph_id": "sg:x",
                        "primary_attack_type": "data_exfiltration",
                        "nodes": [
                            {"node_id": "node:s-src", "role": "source"},
                            {"node_id": "node:s-sink", "role": "sink"},
                        ],
                    }
                ],
            )

            def partial_audit(skill_root: Path, _repo_root: Path) -> AuditReport:
                return _make_report(skill_root.name, positive=skill_root.name == "s-src")

            report = evaluate_blue_on_split(
                split_path=split_path,
                skills_root=skills_root,
                audit_fn=partial_audit,
            )

        self.assertEqual(report.subgraph_recall, 0.0)
        self.assertEqual(report.per_family[0].detected, 0)
        self.assertEqual(report.per_family[0].total, 1)

    def test_missing_skill_directory_marks_subgraph_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skills_root = tmp_path / "skills"
            skills_root.mkdir()
            split_path = tmp_path / "eval.jsonl"
            _write_split(
                split_path,
                [
                    {
                        "subgraph_id": "sg:ghost",
                        "primary_attack_type": "noop",
                        "nodes": [{"node_id": "node:ghost", "role": "source"}],
                    }
                ],
            )

            def should_not_run(*_args: object) -> AuditReport:  # pragma: no cover
                raise AssertionError("audit should not be called for missing skill")

            report = evaluate_blue_on_split(
                split_path=split_path,
                skills_root=skills_root,
                audit_fn=should_not_run,
            )

        self.assertEqual(report.skipped_subgraphs, 1)
        self.assertEqual(report.total_subgraphs, 1)
        self.assertEqual(report.subgraph_recall, 0.0)
        self.assertEqual(len(report.subgraphs), 1)
        self.assertEqual(report.subgraphs[0].missing_skills, ["ghost"])

    def test_render_markdown_includes_family_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skills_root = tmp_path / "skills"
            skill_dir = skills_root / "only-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("#", encoding="utf-8")
            split_path = tmp_path / "eval.jsonl"
            _write_split(
                split_path,
                [
                    {
                        "subgraph_id": "sg:only",
                        "primary_attack_type": "credential_abuse",
                        "nodes": [{"node_id": "node:only-skill", "role": "source"}],
                    }
                ],
            )
            report = evaluate_blue_on_split(
                split_path=split_path,
                skills_root=skills_root,
                audit_fn=lambda *_: _make_report("only-skill", positive=True),
            )

        text = render_markdown(report)
        self.assertIn("# Blue Eval Report", text)
        self.assertIn("credential_abuse", text)
        self.assertIn("1.0000", text)


if __name__ == "__main__":
    unittest.main()
