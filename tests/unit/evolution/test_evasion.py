"""Unit tests for :mod:`red_team.evolution.evasion` (IMPL-09).

``step_s1`` is a thin forwarder to :func:`red_team.agent.runtime.agent_run`.
These tests verify:

1. It invokes ``agent_run`` exactly once with ``search_space="evasion"`` and
   propagates ``round_index``.
2. It propagates exceptions raised by ``agent_run`` to the caller.
3. It honors ``run_config.memory_enabled = False`` by passing
   ``memory_enabled=False`` through.
4. It forwards ``attack_log_path`` verbatim.

All tests use a ``unittest.mock.Mock`` for ``agent_run`` so the runtime
is never actually exercised.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.models import RepositorySlice
from core.schemas.attack_objective import AttackObjective, Channel, Topology
from core.schemas.skill_variant import SkillVariant
from red_team.baselines.base import AgentRunResult
from red_team.campaign.run_config import RunConfig
from red_team.evolution.evasion import step_s1


def make_parent() -> SkillVariant:
    """Canonical S1-test parent: single-skill PERSISTENCE attack."""
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={
            "entry_script": "run.sh",
            "deps": ["requests"],
            "declared_apis": ["http.get"],
        },
        kappa=[Channel.FILESYSTEM, Channel.SUBPROCESS],
        d="# original skill doc",
        chain=["skill-alpha"],
        metadata={},
    )


def make_repo() -> RepositorySlice:
    return RepositorySlice(
        repo_id="repo-test",
        repo_root="/tmp/repo-test",
        readme_path="README.md",
        readme_content="# Test repo",
        file_tree=["README.md"],
        dependency_files=[],
    )


def make_agent_result() -> AgentRunResult:
    return AgentRunResult(
        mutation=None,
        tool_chain=[],
        rationale="stub",
        cost_usd=0.0,
        tokens_in=0,
        tokens_out=0,
        refusal=False,
        stop_reason="submitted",
    )


class StepS1Tests(unittest.TestCase):
    def test_step_s1_forwards_to_agent_run(self) -> None:
        """step_s1 calls agent_run exactly once with search_space='evasion'."""
        parent = make_parent()
        repo = make_repo()
        expected = make_agent_result()
        fake_agent_run = Mock(return_value=expected)
        rc = RunConfig.from_ablation({})

        result = step_s1(
            parent,
            [],
            repo,
            agent_run=fake_agent_run,
            run_config=rc,
            round_index=3,
        )

        self.assertIs(result, expected)
        self.assertEqual(fake_agent_run.call_count, 1)
        _, kwargs = fake_agent_run.call_args
        self.assertEqual(kwargs["search_space"], "evasion")
        self.assertEqual(kwargs["round_index"], 3)

    def test_step_s1_propagates_exception(self) -> None:
        """Exceptions raised by agent_run must bubble up unchanged."""
        parent = make_parent()
        repo = make_repo()
        fake_agent_run = Mock(side_effect=RuntimeError("boom"))
        rc = RunConfig.from_ablation({})

        with self.assertRaises(RuntimeError) as ctx:
            step_s1(
                parent,
                [],
                repo,
                agent_run=fake_agent_run,
                run_config=rc,
                round_index=0,
            )
        self.assertEqual(str(ctx.exception), "boom")

    def test_step_s1_honors_memory_disabled(self) -> None:
        """run_config with memory_enabled=False forwards memory_enabled=False."""
        parent = make_parent()
        repo = make_repo()
        fake_agent_run = Mock(return_value=make_agent_result())
        rc = RunConfig.from_ablation({"memory_enabled": False})

        step_s1(
            parent,
            [],
            repo,
            agent_run=fake_agent_run,
            run_config=rc,
            round_index=0,
        )

        _, kwargs = fake_agent_run.call_args
        self.assertIs(kwargs["memory_enabled"], False)

    def test_step_s1_memory_enabled_default_true(self) -> None:
        """Default RunConfig has memory_enabled=True."""
        parent = make_parent()
        repo = make_repo()
        fake_agent_run = Mock(return_value=make_agent_result())
        rc = RunConfig.from_ablation({})

        step_s1(
            parent,
            [],
            repo,
            agent_run=fake_agent_run,
            run_config=rc,
            round_index=0,
        )

        _, kwargs = fake_agent_run.call_args
        self.assertIs(kwargs["memory_enabled"], True)

    def test_step_s1_passes_attack_log_path(self) -> None:
        """attack_log_path is forwarded verbatim."""
        parent = make_parent()
        repo = make_repo()
        fake_agent_run = Mock(return_value=make_agent_result())
        log_path = Path("/tmp/red-team/attack.log")
        rc = RunConfig.from_ablation({})

        step_s1(
            parent,
            [],
            repo,
            agent_run=fake_agent_run,
            run_config=rc,
            round_index=0,
            attack_log_path=log_path,
        )

        _, kwargs = fake_agent_run.call_args
        self.assertEqual(kwargs["attack_log_path"], log_path)

    def test_step_s1_forwards_run_config(self) -> None:
        """run_config is forwarded so downstream toggles still apply."""
        parent = make_parent()
        repo = make_repo()
        fake_agent_run = Mock(return_value=make_agent_result())
        rc = RunConfig.from_ablation({"tool_use_off": True, "disable_tool": "read_skill"})

        step_s1(
            parent,
            [],
            repo,
            agent_run=fake_agent_run,
            run_config=rc,
            round_index=7,
        )

        _, kwargs = fake_agent_run.call_args
        self.assertIs(kwargs["run_config"], rc)


if __name__ == "__main__":
    unittest.main()
