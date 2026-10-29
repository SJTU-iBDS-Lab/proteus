"""Tests for the multi-skill chain runner."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sandbox.chain_runner import ChainResult, run_chain
from sandbox.openclaw_runner import SandboxConfig


def _completed(
    stdout: str = "", stderr: str = "", rc: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=rc, stdout=stdout, stderr=stderr
    )


class ChainResultTests(unittest.TestCase):
    def test_to_dict(self) -> None:
        r = ChainResult(
            subgraph_id="sg:chain:a>b>c",
            chain=["a", "b", "c"],
            primary_attack_type="data_exfiltration",
            exit_code=0,
            attack_succeeded=True,
        )
        d = r.to_dict()
        self.assertEqual(d["subgraph_id"], "sg:chain:a>b>c")
        self.assertEqual(d["chain"], ["a", "b", "c"])
        self.assertTrue(d["attack_succeeded"])
        self.assertIsInstance(json.dumps(d), str)


class RunChainTests(unittest.TestCase):
    @patch("sandbox.chain_runner.subprocess")
    @patch("sandbox.chain_runner._exec_in_container")
    @patch("sandbox.chain_runner._copy_to_container")
    @patch("sandbox.chain_runner.plant_canaries")
    @patch("sandbox.chain_runner.check_canaries")
    def test_run_chain_basic(
        self,
        mock_check: MagicMock,
        mock_plant: MagicMock,
        mock_copy: MagicMock,
        mock_exec: MagicMock,
        mock_subproc_mod: MagicMock,
    ) -> None:
        from sandbox.honeypot import HoneypotResult

        # Mock all docker interactions
        mock_exec.return_value = _completed()
        mock_copy.return_value = None
        mock_plant.return_value = {}
        mock_check.return_value = HoneypotResult(
            canary_token_leaked=True,
            attack_succeeded=True,
            evidence=["canary_token_in_output"],
        )
        mock_subproc_mod.run.return_value = _completed(
            stdout='{"result": "ok"}\n---EXITCODE:0'
        )
        mock_subproc_mod.TimeoutExpired = subprocess.TimeoutExpired

        with tempfile.TemporaryDirectory() as tmp:
            skill_a = Path(tmp) / "skill-a"
            skill_b = Path(tmp) / "skill-b"
            skill_a.mkdir()
            skill_b.mkdir()
            (skill_a / "SKILL.md").write_text("# Skill A\n")
            (skill_b / "SKILL.md").write_text("# Skill B\n")

            turns = [
                {"phase": "setup", "target_skill": "skill-a", "prompt": "list files"},
                {"phase": "exploit", "target_skill": "skill-b", "prompt": "send data"},
            ]
            config = SandboxConfig(container_name="test-ctr", timeout=60)

            result = run_chain(
                [skill_a, skill_b],
                turns,
                config,
                subgraph_id="sg:chain:a>b",
                primary_attack_type="data_exfiltration",
            )

        self.assertIsInstance(result, ChainResult)
        self.assertEqual(result.chain, ["skill-a", "skill-b"])
        self.assertTrue(result.attack_succeeded)
        self.assertEqual(result.exit_code, 0)

    @patch("sandbox.chain_runner.subprocess")
    @patch("sandbox.chain_runner._exec_in_container")
    @patch("sandbox.chain_runner._copy_to_container")
    @patch("sandbox.chain_runner.plant_canaries")
    @patch("sandbox.chain_runner.check_canaries")
    def test_run_chain_timeout(
        self,
        mock_check: MagicMock,
        mock_plant: MagicMock,
        mock_copy: MagicMock,
        mock_exec: MagicMock,
        mock_subproc_mod: MagicMock,
    ) -> None:
        from sandbox.honeypot import HoneypotResult

        mock_exec.return_value = _completed()
        mock_copy.return_value = None
        mock_plant.return_value = {}
        mock_check.return_value = HoneypotResult()
        mock_subproc_mod.run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=60)
        mock_subproc_mod.TimeoutExpired = subprocess.TimeoutExpired
        mock_subproc_mod.CompletedProcess = subprocess.CompletedProcess

        with tempfile.TemporaryDirectory() as tmp:
            skill_a = Path(tmp) / "skill-a"
            skill_a.mkdir()
            (skill_a / "SKILL.md").write_text("#")

            result = run_chain(
                [skill_a],
                [{"phase": "exploit", "target_skill": "skill-a", "prompt": "bad"}],
                SandboxConfig(timeout=60),
                subgraph_id="sg:timeout",
            )

        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, 124)


if __name__ == "__main__":
    unittest.main()
