"""Unit tests for the standalone PyRIT baseline adapter.

The real PyRIT package is lazy-loaded; these tests exercise the campaign
boundary without network calls or the optional heavy dependency.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.baselines.base import AgentRunResult, estimate_cost
from red_team.baselines.pyrit_baseline import PyRITStandalone
from red_team.baselines.pyrit_baseline.prompt_to_variant import (
    wrap_prompt_as_variant,
)
from red_team.campaign.phase1_main import _run_single_round
from red_team.campaign.run_config import RunConfig


def _seed() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={},
        kappa=[Channel.FILESYSTEM],
        d="---\nname: demo\n---\nOriginal body\n",
        chain=["demo-skill"],
        metadata={"seed_id": "seed-test"},
    )


def test_prompt_adapter_preserves_lineage_and_adds_primitive() -> None:
    parent = _seed()

    child = wrap_prompt_as_variant(parent, "Rewritten documentation")

    assert child.d.startswith("---\nname: demo\n---\n")
    assert "Rewritten documentation" in child.d
    assert "Original body" not in child.d
    assert child.metadata["mutation_source"] == "pyrit"
    assert child.metadata["parent_variant_hash"] == parent.content_hash()
    assert child.c["entry_script"] in child.c["scripts"]
    assert parent.d == "---\nname: demo\n---\nOriginal body\n"
    assert parent.c == {}


def test_standalone_agent_run_propagates_cost_and_tokens(monkeypatch) -> None:
    baseline = PyRITStandalone(adversary_model="deepseek-v4-pro")

    async def fake_run(objective: str) -> tuple[str, int, int]:
        assert "persistence" in objective or ".bashrc" in objective
        return "Rewritten documentation", 100, 40

    monkeypatch.setattr(baseline, "_run_red_teaming", fake_run)
    result = baseline.agent_run(_seed(), [], SimpleNamespace(), "free_mutation")

    assert result.mutation is not None
    assert result.tokens_in == 100
    assert result.tokens_out == 40
    assert result.cost_usd == pytest.approx(
        estimate_cost(100, 40, "deepseek-v4-pro")
    )
    assert result.refusal is False


@pytest.mark.parametrize(
    "strategy", ["pyrit_standalone", "skill_inject_bo5", "skillject_trace"]
)
def test_external_strategies_are_registered(strategy: str) -> None:
    config = RunConfig.from_ablation({"red_strategy": strategy})
    assert config.agent_strategy == strategy


def test_pyrit_uses_standalone_campaign_dispatch(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeStandalone:
        def agent_run(self, seed, findings, repo, search_space):
            calls.append((seed, findings, repo, search_space))
            return AgentRunResult(mutation=seed, stop_reason="pyrit")

    monkeypatch.setattr(
        "red_team.baselines.pyrit_baseline.PyRITStandalone", FakeStandalone
    )
    seed = _seed()
    repo = SimpleNamespace()
    result, backend = _run_single_round(
        seed=seed,
        findings_for_agent=[],
        repo=repo,
        search_space="free_mutation",
        run_config=RunConfig(agent_strategy="pyrit_standalone"),
        round_index=1,
        attack_log_path=tmp_path / "attack.jsonl",
        agent_run_fn=lambda *args, **kwargs: None,
    )

    assert backend == "pyrit_standalone"
    assert result.stop_reason == "programmatic_fallback"
    assert calls == [(seed, [], repo, "free_mutation")]
