"""Tests for the OpenClaw sandbox runner with mocked subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sandbox.openclaw_runner import (
    SandboxConfig,
    SandboxResult,
    _parse_tool_trace,
    preflight,
    run_seed,
)


def _completed(stdout: str = "", stderr: str = "", rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


class PreflightTests(unittest.TestCase):
    @patch("sandbox.openclaw_runner.subprocess.run")
    def test_preflight_ok(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _completed(stdout="24.0.0"),
            _completed(stdout="true"),
        ]
        config = SandboxConfig(container_name="test-container")
        result = preflight(config)
        self.assertTrue(result["ok"])

    @patch("sandbox.openclaw_runner.subprocess.run")
    def test_preflight_docker_unavailable(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _completed(stderr="Cannot connect", rc=1)
        result = preflight(SandboxConfig())
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "docker_unavailable")

    @patch("sandbox.openclaw_runner.subprocess.run")
    def test_preflight_container_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _completed(stdout="24.0.0"),
            _completed(stderr="No such container", rc=1),
        ]
        result = preflight(SandboxConfig())
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "container_not_found")

    @patch("sandbox.openclaw_runner.subprocess.run")
    def test_preflight_starts_stopped_container(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _completed(stdout="24.0.0"),
            _completed(stdout="false"),
            _completed(stdout="test-container"),
        ]
        result = preflight(SandboxConfig(container_name="test-container"))
        self.assertTrue(result["ok"])


class ParseToolTraceTests(unittest.TestCase):
    def test_parses_tool_events(self) -> None:
        log = (
            "[agent/embedded] embedded run tool start: runId=r1 tool=Bash\n"
            "[agent/embedded] embedded run tool end: runId=r1 tool=Bash\n"
            "[agent/embedded] embedded run tool start: runId=r2 tool=Read\n"
            "[agent/embedded] embedded run tool end: runId=r2 tool=Read\n"
        )
        trajectory, calls, obs = _parse_tool_trace(log)
        self.assertEqual(len(trajectory), 4)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["tool"], "Bash")
        self.assertEqual(calls[1]["tool"], "Read")
        self.assertEqual(obs["tool_call_count"], 2)

    def test_empty_input(self) -> None:
        trajectory, calls, obs = _parse_tool_trace("")
        self.assertEqual(trajectory, [])
        self.assertEqual(calls, [])
        self.assertEqual(obs, {})


class RunSeedTests(unittest.TestCase):
    @patch("sandbox.openclaw_runner.subprocess.run")
    def test_run_seed_captures_output(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _completed(),  # rm + mkdir
            _completed(),  # docker cp skill
            _completed(),  # docker cp turns
            _completed(stdout='{"result": "ok"}\n---EXITCODE:0'),  # exec
            _completed(stdout=""),  # cat stderr log
        ]

        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Test\n")

            config = SandboxConfig(container_name="test-ctr")
            turns = [{"phase": "setup", "prompt": "hello"}]
            result = run_seed(
                skill_dir, turns, config, subgraph_id="sg:test"
            )

        self.assertIsInstance(result, SandboxResult)
        self.assertEqual(result.subgraph_id, "sg:test")
        self.assertEqual(result.skill_id, "test-skill")
        self.assertEqual(result.exit_code, 0)
        self.assertFalse(result.timed_out)

    @patch("sandbox.openclaw_runner.subprocess.run")
    def test_run_seed_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _completed(),
            _completed(),
            _completed(),
            _completed(stdout="---EXITCODE:0", rc=124, stderr="timeout after 300s"),
            _completed(stdout=""),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("#")

            result = run_seed(
                skill_dir,
                [{"phase": "exploit", "prompt": "do bad things"}],
                SandboxConfig(),
                subgraph_id="sg:timeout",
            )

        self.assertTrue(result.timed_out)

    def test_sandbox_result_to_dict(self) -> None:
        r = SandboxResult(subgraph_id="sg:1", skill_id="s1", exit_code=0)
        d = r.to_dict()
        self.assertEqual(d["subgraph_id"], "sg:1")
        self.assertIsInstance(json.dumps(d), str)


if __name__ == "__main__":
    unittest.main()
