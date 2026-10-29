"""Lethal-gated audit mode (audit_gate="lethal"): the blue team must run
only after a variant proves lethal in the sandbox, never before."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from red_team.campaign import phase1_main as pm
from red_team.campaign.io import _RunPaths
from red_team.campaign.run_config import RunConfig
from red_team.campaign.state import _RunState
from core.schemas import (
    AttackObjective,
    Channel,
    SkillVariant,
    Topology,
)
from sandbox.honeypot import HoneypotResult


def _variant(tag: str) -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=[Channel.FILESYSTEM],
        d=f"doc::{tag}",
        chain=["skill-alpha"],
        metadata={"tag": tag},
    )


def _paths(tmp_path: Path) -> _RunPaths:
    (tmp_path / "mutations").mkdir()
    (tmp_path / "chains").mkdir()
    return _RunPaths(
        run_dir=tmp_path,
        log=tmp_path / "log.jsonl",
        attack_log=tmp_path / "attack_log.jsonl",
        summary=tmp_path / "summary.json",
        mutations_dir=tmp_path / "mutations",
        chains_dir=tmp_path / "chains",
        harvest=tmp_path / "harvest.jsonl",
        memory_db=tmp_path / "memory.sqlite",
    )


def _run_round(monkeypatch, tmp_path, *, round_ix, state, lethal, blue_calls, **cfg_kw):
    cfg = RunConfig(phase="main_only", memory_mode="off", audit_gate="lethal", **cfg_kw)
    child = _variant(f"child-r{round_ix}")

    def fake_blue(*a, **k):
        blue_calls.append(1)
        return []

    monkeypatch.setattr(pm._seams, "call_blue_team", fake_blue)
    monkeypatch.setattr(pm._io, "persist_mutation_md", lambda *a, **k: None)
    monkeypatch.setattr(pm._io, "snapshot_chain", lambda *a, **k: None)
    monkeypatch.setattr(pm, "extract_target_reply", lambda *a, **k: "")
    monkeypatch.setattr(pm, "synthesize_runtime_feedback", lambda *a, **k: [])
    monkeypatch.setattr(
        pm, "_run_single_round",
        lambda **k: (SimpleNamespace(
            mutation=child, cost_usd=0.0, stop_reason="submitted",
            rationale="", tool_chain=[],
        ), "stub"),
    )

    return pm._run_one_round(
        round_ix,
        state=state, paths=_paths(tmp_path), run_config=cfg,
        search_space="free_mutation", rounds=5,
        blue_team="aig", blue_team_fn=None,
        sandbox_container="c0", target_model="glm-5",
        agent_run_fn=None,
        oracle_fn=lambda *a, **k: lethal,
        sandbox_run_fn=lambda v, chain, model, container: (
            HoneypotResult(), [], None
        ),
        seed=state.current_seed,
        repo=SimpleNamespace(repo_id="repo"),
        run_id="rid", seed_id="sid",
        memory_runtime=SimpleNamespace(enabled=False),
    )


def test_gate_skips_audit_when_not_lethal(monkeypatch, tmp_path):
    state = _RunState(current_seed=_variant("parent"))
    calls: list[int] = []
    star = _run_round(monkeypatch, tmp_path, round_ix=1, state=state,
                      lethal=False, blue_calls=calls)
    assert calls == []                    # blue team never invoked
    assert star is False
    assert state.last_audit_findings == []
    # never-audited rounds must not count as bypass (empty findings
    # would otherwise read as APPROVE)
    assert state.total_bypass == 0


def test_gate_audits_after_sandbox_when_lethal(monkeypatch, tmp_path):
    state = _RunState(current_seed=_variant("parent"))
    calls: list[int] = []
    star = _run_round(monkeypatch, tmp_path, round_ix=1, state=state,
                      lethal=True, blue_calls=calls)
    assert calls == [1]                   # exactly one deferred audit
    assert star is True                   # [] findings => bypass; lethal => ★
    assert state.total_bypass == 1 and state.total_lethal == 1


def test_forced_single_topology_audits_the_executed_variant(monkeypatch, tmp_path):
    """force_topology=single: the deferred audit (and next-round parent)
    must see the truncated single-skill variant the sandbox executed,
    not the pre-rewrite pair chain."""
    state = _RunState(current_seed=_variant("parent"))

    from dataclasses import replace as _dc_replace
    child = _dc_replace(_variant("child-pair"), tau=Topology.PAIR,
                        chain=["skill-a", "skill-b"])
    monkeypatch.setattr(
        pm, "_run_single_round",
        lambda **k: (SimpleNamespace(
            mutation=child, cost_usd=0.0, stop_reason="submitted",
            rationale="", tool_chain=[],
        ), "stub"),
    )
    audited: list = []

    def fake_blue(variant, *a, **k):
        audited.append(variant)
        return []

    monkeypatch.setattr(pm._seams, "call_blue_team", fake_blue)
    monkeypatch.setattr(pm._io, "persist_mutation_md", lambda *a, **k: None)
    monkeypatch.setattr(pm._io, "snapshot_chain", lambda *a, **k: None)
    monkeypatch.setattr(pm, "extract_target_reply", lambda *a, **k: "")
    monkeypatch.setattr(pm, "synthesize_runtime_feedback", lambda *a, **k: [])

    cfg = RunConfig(phase="main_only", memory_mode="off",
                    audit_gate="lethal", topology_constraint="single")
    star = pm._run_one_round(
        1,
        state=state, paths=_paths(tmp_path), run_config=cfg,
        search_space="free_mutation", rounds=5,
        blue_team="aig", blue_team_fn=None,
        sandbox_container="c0", target_model="glm-5",
        agent_run_fn=None,
        oracle_fn=lambda *a, **k: True,
        sandbox_run_fn=lambda v, chain, model, container: (
            HoneypotResult(), [], None
        ),
        seed=state.current_seed,
        repo=SimpleNamespace(repo_id="repo"),
        run_id="rid", seed_id="sid",
        memory_runtime=SimpleNamespace(enabled=False),
    )
    assert star is True
    assert len(audited) == 1
    assert list(audited[0].chain) == ["skill-a"]   # not ["skill-a","skill-b"]
    assert list(state.current_seed.chain) == ["skill-a"]
