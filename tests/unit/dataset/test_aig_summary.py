"""Tests for A.I.G scan summary aggregation."""

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

from dataset.aig_analyzer import AIGFinding, AIGResult
from dataset.aig_summary import render_markdown, summarize, write_results


def _result(skill_id: str, findings: list[AIGFinding]) -> AIGResult:
    return AIGResult(skill_id=skill_id, findings=findings)


class SummarizeTests(unittest.TestCase):
    def test_summary_aggregates_levels_and_risk_types(self) -> None:
        results = [
            _result(
                "s1",
                [
                    AIGFinding(title="a", level="high", risk_type="command_injection"),
                    AIGFinding(title="b", level="low", risk_type="data_exfiltration"),
                ],
            ),
            _result("s2", []),
            _result(
                "s3",
                [AIGFinding(title="c", level="high", risk_type="command_injection")],
            ),
        ]
        summary = summarize(results)
        self.assertEqual(summary.scanned, 3)
        self.assertEqual(summary.skills_with_findings, 2)
        self.assertEqual(summary.total_findings, 3)
        self.assertEqual(summary.by_level["high"], 2)
        self.assertEqual(summary.by_level["low"], 1)
        self.assertEqual(summary.by_risk_type["command_injection"], 2)

    def test_render_markdown_contains_sections(self) -> None:
        summary = summarize(
            [_result("s", [AIGFinding(title="t", level="medium", risk_type="noop")])]
        )
        text = render_markdown(summary, error_count=1)
        self.assertIn("A.I.G Deep Scan Summary", text)
        self.assertIn("Errors: 1", text)
        self.assertIn("medium", text)
        self.assertIn("noop", text)

    def test_write_results_emits_jsonl_md_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jsonl_path, md_path, errors_path = write_results(
                results=[_result("only", [AIGFinding(title="t", level="high")])],
                errors=[{"skill_id": "ghost", "error": "missing"}],
                output_dir=tmp_path,
            )
            self.assertTrue(jsonl_path.exists())
            self.assertTrue(md_path.exists())
            self.assertTrue(errors_path.exists())
            payload = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(payload["skill_id"], "only")
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Errors: 1", md_text)
            errors_text = errors_path.read_text(encoding="utf-8")
            self.assertIn("ghost", errors_text)

    def test_write_results_removes_stale_error_file_when_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stale = tmp_path / "aig_errors.jsonl"
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("old\n", encoding="utf-8")
            _, _, errors_path = write_results(results=[], errors=[], output_dir=tmp_path)
            self.assertFalse(errors_path.exists())


if __name__ == "__main__":
    unittest.main()
