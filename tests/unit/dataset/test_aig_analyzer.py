"""Tests for the A.I.G mcp_scan adapter with mocked HTTP calls."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.aig_analyzer import AIGFinding, AIGResult, scan_skill


def _mock_urlopen_factory(
    upload_file_url: str = "https://aig.example/files/skill.zip",
    session_id: str = "sess-abc",
    status_sequence: list[str] | None = None,
    result_items: list[dict[str, str]] | None = None,
):
    """Return a side_effect function that simulates AIG server responses."""
    if status_sequence is None:
        status_sequence = ["running", "done"]
    if result_items is None:
        result_items = [
            {
                "title": "Command injection via os.system",
                "description": "Unsanitized input passed to os.system",
                "risk_type": "command_injection",
                "level": "high",
                "suggestion": "Use subprocess with shell=False",
            }
        ]

    call_index = {"status": 0}

    def _urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        method = req.get_method() if hasattr(req, "get_method") else "GET"

        if "/upload" in url:
            body = json.dumps({"data": {"fileUrl": upload_file_url}}).encode()
        elif "/tasks" in url and method == "POST":
            body = json.dumps({"data": {"session_id": session_id}}).encode()
        elif "/status/" in url:
            idx = min(call_index["status"], len(status_sequence) - 1)
            status_val = status_sequence[idx]
            call_index["status"] += 1
            body = json.dumps({"data": {"status": status_val}}).encode()
        elif "/result/" in url:
            body = json.dumps(
                {"data": {"result": {"results": result_items, "readme": "# Skill"}}}
            ).encode()
        else:
            body = b"{}"

        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    return _urlopen


class ScanSkillTests(unittest.TestCase):
    def _make_skill_dir(self, tmp: Path, skill_id: str = "test-skill") -> Path:
        skill_dir = tmp / skill_id
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
        (skill_dir / "main.py").write_text("print('hello')\n", encoding="utf-8")
        return skill_dir

    @patch("dataset.aig_analyzer.urllib_request.urlopen")
    @patch("dataset.aig_analyzer.time.sleep", return_value=None)
    def test_scan_skill_returns_findings(self, _sleep: MagicMock, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _mock_urlopen_factory()
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = self._make_skill_dir(Path(tmp))
            result = scan_skill(
                skill_dir,
                base_url="http://fake:18088",
                model="test-model",
                api_key="test-key",
                model_base_url="http://fake-llm",
            )

        self.assertIsInstance(result, AIGResult)
        self.assertEqual(result.skill_id, "test-skill")
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].risk_type, "command_injection")
        self.assertEqual(result.findings[0].level, "high")
        self.assertFalse(result.cached)
        self.assertEqual(result.session_id, "sess-abc")

    @patch("dataset.aig_analyzer.urllib_request.urlopen")
    @patch("dataset.aig_analyzer.time.sleep", return_value=None)
    def test_scan_skill_caches_result(self, _sleep: MagicMock, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _mock_urlopen_factory()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skill_dir = self._make_skill_dir(tmp_path)
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()

            result1 = scan_skill(
                skill_dir,
                base_url="http://fake:18088",
                model="m",
                api_key="k",
                model_base_url="http://x",
                cache_dir=cache_dir,
            )
            self.assertFalse(result1.cached)
            self.assertTrue((cache_dir / "test-skill.json").exists())

            mock_urlopen.reset_mock()
            result2 = scan_skill(
                skill_dir,
                base_url="http://fake:18088",
                model="m",
                api_key="k",
                model_base_url="http://x",
                cache_dir=cache_dir,
            )
            self.assertTrue(result2.cached)
            self.assertEqual(len(result2.findings), 1)
            mock_urlopen.assert_not_called()

    @patch("dataset.aig_analyzer.urllib_request.urlopen")
    @patch("dataset.aig_analyzer.time.sleep", return_value=None)
    def test_scan_skill_empty_findings(self, _sleep: MagicMock, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _mock_urlopen_factory(result_items=[])
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = self._make_skill_dir(Path(tmp))
            result = scan_skill(
                skill_dir,
                base_url="http://fake:18088",
                model="m",
                api_key="k",
                model_base_url="http://x",
            )
        self.assertEqual(result.findings, [])

    @patch("dataset.aig_analyzer.urllib_request.urlopen")
    @patch("dataset.aig_analyzer.time.sleep", return_value=None)
    def test_scan_skill_task_failure_raises(self, _sleep: MagicMock, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _mock_urlopen_factory(status_sequence=["failed"])
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = self._make_skill_dir(Path(tmp))
            with self.assertRaises(RuntimeError):
                scan_skill(
                    skill_dir,
                    base_url="http://fake:18088",
                    model="m",
                    api_key="k",
                    model_base_url="http://x",
                )

    @patch("dataset.aig_analyzer.urllib_request.urlopen")
    def test_scan_skill_timeout_raises(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _mock_urlopen_factory(
            status_sequence=["running"] * 100
        )
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = self._make_skill_dir(Path(tmp))
            with self.assertRaises(TimeoutError):
                scan_skill(
                    skill_dir,
                    base_url="http://fake:18088",
                    model="m",
                    api_key="k",
                    model_base_url="http://x",
                    poll_timeout=0.01,
                    poll_interval=0.001,
                )


class AIGResultTests(unittest.TestCase):
    def test_to_dict_roundtrip(self) -> None:
        result = AIGResult(
            skill_id="s1",
            findings=[AIGFinding(title="t", level="high", risk_type="xss")],
            readme="# R",
            session_id="sess-1",
        )
        d = result.to_dict()
        self.assertEqual(d["skill_id"], "s1")
        self.assertEqual(len(d["findings"]), 1)
        self.assertEqual(d["findings"][0]["risk_type"], "xss")


if __name__ == "__main__":
    unittest.main()
