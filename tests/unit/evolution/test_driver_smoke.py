"""Smoke tests for :mod:`red_team.evolution.driver` (IMPL-13).

Every test uses the injection seams (``blue_team_fn`` / ``sandbox_run_fn``
/ ``agent_run_fn`` / ``oracle_fn`` / ``horizontal_expand_fn``) so no real
LLM or docker sandbox is invoked.  The tests cover the seven scenarios
enumerated in the task spec:

1. Three-round run with bypass+lethal at round 2 → expansion rows emitted.
2. ablation A5 (``disable_expansion``) suppresses expansion rows.
3. ablation A9 (``feedback="scalar"``) zeros out findings at the agent.
4. ablation A4 (``tool_use_off``) swaps in the random baseline.
5. ablation A7 (``force_topology="single"``) truncates the chain for sandbox.
6. Terminal ``summary.json`` is written with correct keys.
7. Refusal round (``mutation=None``) continues the loop without sandbox.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.models import RepositorySlice
from core.schemas import (
    AttackObjective,
    Channel,
    Evidence,
    PaperFinding,
    SkillVariant,
    Topology,
)
from red_team.baselines.base import AgentRunResult
from red_team.evolution import driver as drv
from red_team.evolution.horizontal_expand import HorizontalExpansion
from red_team.evolution.path_expansion import PathExpandResult
from red_team.evolution.surface_expansion import SurfaceFunnel, SurfaceResult
from sandbox.honeypot import HoneypotResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _seed_single() -> SkillVariant:
    """Single-skill seed (τ = single, chain length 1)."""
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=[Channel.FILESYSTEM],
        d="seed doc",
        chain=["skill-alpha"],
        metadata={"seed_id": "seed-001"},
    )


def _seed_pair() -> SkillVariant:
    """Two-skill seed (τ = pair, chain length 2) for A7 test."""
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.PAIR,
        c={"entry_script": "run.sh"},
        kappa=[Channel.FILESYSTEM, Channel.SUBPROCESS],
        d="pair seed doc",
        chain=["skill-alpha", "skill-beta"],
        metadata={"seed_id": "seed-pair-001"},
    )


def _repo() -> RepositorySlice:
    return RepositorySlice(
        repo_id="r1",
        repo_root="/tmp/r1",
        readme_path=None,
        readme_content="",
        file_tree=[],
        dependency_files=[],
    )


def _mutation_of(parent: SkillVariant, tag: str) -> SkillVariant:
    """Clone ``parent`` with a ``d`` suffix — keeps S1 invariants."""
    metadata = dict(parent.metadata)
    metadata["mutation_tag"] = tag
    return SkillVariant(
        o=parent.o,
        tau=parent.tau,
        c=dict(parent.c) if parent.c else {},
        kappa=list(parent.kappa),
        d=f"{parent.d}::{tag}",
        poison_index=parent.poison_index,
        chain=list(parent.chain),
        metadata=metadata,
    )


def _finding(sev: str) -> PaperFinding:
    return PaperFinding(
        finding_id=f"A-{sev}",
        layer="A",
        category="guidance_injection",
        reason_code="autonomy_encouragement",
        confidence=0.8,
        severity=sev,
        evidence=Evidence(snippet=""),
        explanation="test",
    )


def _lethal_hp() -> HoneypotResult:
    """HoneypotResult that satisfies the PERSISTENCE oracle rule."""
    return HoneypotResult(
        files_modified=["/root/.bashrc"],
        evidence=["persistence_file_modified:.bashrc"],
    )


def _benign_hp() -> HoneypotResult:
    return HoneypotResult(files_created=["/tmp/harmless"])


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# 1. Smoke: 3 rounds, round 2 is bypass+lethal, triggers expansion.
# ---------------------------------------------------------------------------


def test_driver_smoke_three_rounds_with_expansion(tmp_path):
    seed = _seed_single()
    repo = _repo()

    # Blue team: round 1 returns a critical finding → bypass blocked; round 2+
    # returns an empty list → bypass allowed.
    call_count = {"n": 0}

    def blue_team_fn(variant, r):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [_finding("critical")]
        return []

    # Agent returns a fresh mutation every time (stop_reason='submitted').
    def agent_run_fn(parent, findings, r, **kwargs):
        return AgentRunResult(
            mutation=_mutation_of(parent, f"r{call_count['n']}"),
            tool_chain=[{"tool": "read_skill", "args": {}}],
            rationale="test mutation",
            cost_usd=0.01,
            tokens_in=10,
            tokens_out=20,
            refusal=False,
            stop_reason="submitted",
        )

    # Sandbox: round 1 benign HP (non-lethal), round 2 lethal HP.
    sandbox_calls = {"n": 0}

    def sandbox_run_fn(variant, chain, target_model, container):
        sandbox_calls["n"] += 1
        hp = _benign_hp() if sandbox_calls["n"] == 1 else _lethal_hp()
        return hp, []

    # Oracle: mirror the sandbox call count so round 1 → False, round 2 → True.
    def oracle_fn(variant, hp, o):
        return any("persistence_file_modified" in e for e in hp.evidence)

    # Mock horizontal_expand: produces 1 S2 row + 1 S3 row.  Mirrors
    # the production contract — fires ``on_path_result`` /
    # ``on_surface_executed`` immediately after appending each entry
    # so the runner's per-entry handler audits + flushes them to disk.
    def horizontal_expand_fn(parent, r, **kwargs):
        on_path = kwargs.get("on_path_result")
        on_surface = kwargs.get("on_surface_executed")
        mutation = _mutation_of(parent, "s2")
        s2_row = PathExpandResult(
            mutation=mutation,
            tool_chain=[{"tool": "read_skill"}],
            hp=_lethal_hp(),
            bypassed=None,
            lethal=True,
            family_candidate={"name": "cron-write"},
            score=5.0,
            rationale="S2 path",
        )
        if on_path is not None:
            on_path(s2_row)
        s3_mut = _mutation_of(parent, "s3")
        s3_row = SurfaceResult(
            mutation=s3_mut,
            hp=_lethal_hp(),
            bypassed=None,
            lethal=True,
            o_prime=AttackObjective.DATA_EXFILTRATION,
            shared_motif={"kappa": ["filesystem"]},
            post_hoc_ok=True,
            rationale="S3 transfer",
        )
        if on_surface is not None:
            on_surface(s3_row)
        return HorizontalExpansion(
            path_results=[s2_row],
            surface_funnel=SurfaceFunnel(
                ideated=[], selected=[], executed=[s3_row]
            ),
        )

    run_dir = drv.run_evolution(
        seed,
        repo,
        rounds=3,
        blue_team="ours",
        blue_team_fn=blue_team_fn,
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_run_fn,
        oracle_fn=oracle_fn,
        horizontal_expand_fn=horizontal_expand_fn,
        output_root=tmp_path,
        run_id="smoke",
        # Single-seed smoke test: barrier is trivial (no siblings to
        # wait for, no offline extractor to interleave).
        phase2_barrier=lambda _winner_path: None,
    )

    rows = _read_log(run_dir / "log.jsonl")
    main_rows = [r for r in rows if not r["is_expansion"]]
    exp_rows = [r for r in rows if r["is_expansion"]]

    assert len(main_rows) == 2, f"expected 2 main rows, got {len(main_rows)}"
    assert main_rows[0]["round"] == 1
    assert main_rows[0]["bypassed"] is False
    assert main_rows[0]["lethal"] is False
    assert main_rows[1]["round"] == 2
    assert main_rows[1]["bypassed"] is True
    assert main_rows[1]["lethal"] is True
    assert main_rows[1]["backend"] == "agent"
    assert main_rows[1]["phase"] == "free_mutation"

    # ≥1 expansion row (we asserted 2 above: one path_expansion + one surface_expansion).
    assert len(exp_rows) >= 1
    phases = {r["phase"] for r in exp_rows}
    assert "path_expansion" in phases
    assert "surface_expansion" in phases

    # summary.json was written and is consistent.
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["rounds_ran"] == 2
    assert summary["first_bypass_round"] == 2
    assert summary["total_bypass"] == 1
    assert summary["total_lethal"] == 1


# ---------------------------------------------------------------------------
# 2. A5: disable_expansion ⇒ no expansion rows.
# ---------------------------------------------------------------------------


def test_driver_ablation_disable_expansion(tmp_path):
    seed = _seed_single()
    repo = _repo()

    def blue_team_fn(_v, _r):
        return []  # always bypass

    def agent_run_fn(parent, findings, r, **kwargs):
        return AgentRunResult(
            mutation=_mutation_of(parent, "m"),
            tool_chain=[],
            rationale="",
            cost_usd=0.0,
            stop_reason="submitted",
        )

    def sandbox_run_fn(v, c, t, ct):
        return _lethal_hp(), []

    def oracle_fn(v, hp, o):
        return True

    hx_mock = MagicMock()

    run_dir = drv.run_evolution(
        seed,
        repo,
        rounds=3,
        ablation={"disable_expansion": True},
        blue_team="ours",
        blue_team_fn=blue_team_fn,
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_run_fn,
        oracle_fn=oracle_fn,
        horizontal_expand_fn=hx_mock,
        output_root=tmp_path,
        run_id="no-expand",
    )

    rows = _read_log(run_dir / "log.jsonl")
    assert all(not r["is_expansion"] for r in rows)
    # horizontal_expand is *never* invoked under A5.
    hx_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 3. A9: feedback='scalar' ⇒ agent receives findings=[].
# ---------------------------------------------------------------------------


def test_driver_ablation_feedback_scalar(tmp_path):
    seed = _seed_single()
    repo = _repo()

    def blue_team_fn(_v, _r):
        # Non-empty (critical) — under A9 this must NOT reach the agent.
        return [_finding("critical"), _finding("low")]

    observed_findings: list[list[PaperFinding]] = []

    def agent_run_fn(parent, findings, r, **kwargs):
        observed_findings.append(list(findings))
        return AgentRunResult(
            mutation=_mutation_of(parent, "m"),
            tool_chain=[],
            rationale="",
            cost_usd=0.0,
            stop_reason="submitted",
        )

    drv.run_evolution(
        seed,
        repo,
        rounds=1,
        ablation={"feedback": "scalar"},
        blue_team="ours",
        blue_team_fn=blue_team_fn,
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=lambda *a, **k: (_benign_hp(), []),
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="a9",
    )

    assert observed_findings, "agent was never called"
    for call in observed_findings:
        assert call == [], f"expected empty findings under A9, got {call}"


def test_lethal_audit_gate_audits_only_lethal_candidate_after_sandbox(tmp_path):
    seed = _seed_single()
    repo = _repo()
    events: list[tuple[str, str]] = []
    candidate_tags: list[str] = []
    summaries: list[dict | None] = []

    def agent_run_fn(parent, findings, r, **kwargs):
        summaries.append(kwargs.get("last_round_summary"))
        mutation = _mutation_of(parent, f"r{kwargs['round_index']}")
        candidate_tags.append(str(mutation.metadata["mutation_tag"]))
        return AgentRunResult(
            mutation=mutation,
            tool_chain=[],
            rationale="",
            cost_usd=0.0,
            stop_reason="submitted",
        )

    sandbox_calls = {"n": 0}

    def sandbox_run_fn(variant, chain, target_model, container):
        sandbox_calls["n"] += 1
        tag = str(variant.metadata["mutation_tag"])
        events.append(("sandbox", tag))
        hp = _benign_hp() if sandbox_calls["n"] == 1 else _lethal_hp()
        return hp, []

    def blue_team_fn(variant, _repo):
        tag = str(variant.metadata["mutation_tag"])
        events.append(("audit", tag))
        return []

    run_dir = drv.run_evolution(
        seed,
        repo,
        rounds=2,
        ablation={
            "audit_gate": "lethal",
            "disable_expansion": True,
            "phase": "main_only",
        },
        blue_team="ours",
        blue_team_fn=blue_team_fn,
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_run_fn,
        oracle_fn=lambda _v, hp, _o: any(
            "persistence_file_modified" in e for e in hp.evidence
        ),
        output_root=tmp_path,
        run_id="lethal-gate",
    )

    assert candidate_tags == ["r1", "r2"]
    assert summaries[0] is None
    assert summaries[1] is not None
    assert summaries[1]["audit_verdict"] == "NOT_RUN"
    assert summaries[1]["star_status"] == "no_progress (audit not run)"
    assert events == [
        ("sandbox", "r1"),
        ("sandbox", "r2"),
        ("audit", "r2"),
    ]
    rows = _read_log(run_dir / "log.jsonl")
    assert [(row["lethal"], row["bypassed"]) for row in rows] == [
        (False, False),
        (True, True),
    ]


def test_scalar_lethal_gate_exposes_verdict_without_finding_codes(tmp_path):
    summaries: list[dict | None] = []
    findings_seen: list[list[PaperFinding]] = []

    def agent_run_fn(parent, findings, _repo, **kwargs):
        findings_seen.append(list(findings))
        summaries.append(kwargs.get("last_round_summary"))
        return AgentRunResult(
            mutation=_mutation_of(parent, f"r{kwargs['round_index']}"),
            tool_chain=[],
            rationale="",
            cost_usd=0.0,
            stop_reason="submitted",
        )

    drv.run_evolution(
        _seed_single(),
        _repo(),
        rounds=2,
        ablation={
            "audit_gate": "lethal",
            "feedback": "scalar",
            "disable_expansion": True,
            "phase": "main_only",
        },
        blue_team="ours",
        blue_team_fn=lambda _v, _r: [_finding("critical")],
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=lambda *a, **k: (_lethal_hp(), []),
        oracle_fn=lambda *a, **k: True,
        output_root=tmp_path,
        run_id="scalar-lethal-gate",
    )

    assert findings_seen == [[], []]
    assert summaries[0] is None
    assert summaries[1] is not None
    assert summaries[1]["audit_verdict"] == "REJECT"
    assert summaries[1]["audit_findings"] == []


# ---------------------------------------------------------------------------
# 4. A4: tool_use_off ⇒ random-baseline backend.
# ---------------------------------------------------------------------------


def test_driver_ablation_tool_use_off(tmp_path):
    seed = _seed_single()
    repo = _repo()

    def blue_team_fn(_v, _r):
        return []

    agent_mock = MagicMock()

    run_dir = drv.run_evolution(
        seed,
        repo,
        rounds=1,
        ablation={"tool_use_off": True},
        blue_team="ours",
        blue_team_fn=blue_team_fn,
        # This must NOT be called because tool_use_off bypasses step_s1.
        agent_run_fn=agent_mock,
        sandbox_run_fn=lambda *a, **k: (_benign_hp(), []),
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="a4",
    )

    rows = _read_log(run_dir / "log.jsonl")
    assert rows, "expected at least one log row"
    assert all(r["backend"] == "random_baseline" for r in rows)
    assert all(r["stop_reason"] == "programmatic_fallback" for r in rows)
    agent_mock.assert_not_called()


# ---------------------------------------------------------------------------
# 5. A7: force_topology=single ⇒ sandbox sees truncated chain.
# ---------------------------------------------------------------------------


def test_driver_ablation_force_topology_single(tmp_path):
    seed = _seed_pair()  # τ = pair, chain length 2
    repo = _repo()

    def blue_team_fn(_v, _r):
        return []

    def agent_run_fn(parent, findings, r, **kwargs):
        return AgentRunResult(
            mutation=_mutation_of(parent, "m"),
            tool_chain=[],
            rationale="",
            cost_usd=0.0,
            stop_reason="submitted",
        )

    sandbox_seen: list[SkillVariant] = []

    def sandbox_run_fn(variant, chain, target_model, container):
        sandbox_seen.append(variant)
        return _benign_hp(), []

    run_dir = drv.run_evolution(
        seed,
        repo,
        rounds=1,
        ablation={"force_topology": "single"},
        blue_team="ours",
        blue_team_fn=blue_team_fn,
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_run_fn,
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="a7",
    )

    assert len(sandbox_seen) == 1
    forced_variant = sandbox_seen[0]
    assert len(forced_variant.chain) == 1
    assert forced_variant.tau == Topology.SINGLE

    rows = _read_log(run_dir / "log.jsonl")
    assert rows[0]["forced_topology"] is True
    assert rows[0]["chain_original"] == ["skill-alpha", "skill-beta"]
    assert rows[0]["chain_used"] == ["skill-alpha"]


# ---------------------------------------------------------------------------
# 6. summary.json is written with expected keys.
# ---------------------------------------------------------------------------


def test_driver_writes_summary_json(tmp_path):
    seed = _seed_single()
    repo = _repo()

    def blue_team_fn(_v, _r):
        return [_finding("critical")]  # bypass never satisfied

    def agent_run_fn(parent, findings, r, **kwargs):
        return AgentRunResult(
            mutation=_mutation_of(parent, "x"),
            tool_chain=[],
            rationale="",
            cost_usd=0.05,
            stop_reason="submitted",
        )

    run_dir = drv.run_evolution(
        seed,
        repo,
        rounds=1,
        blue_team="ours",
        blue_team_fn=blue_team_fn,
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=lambda *a, **k: (_benign_hp(), []),
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="summary-test",
        target_model="glm-5",
        mutator_model="claude-sonnet-4-6",
    )

    summary_path = run_dir / "summary.json"
    assert summary_path.exists(), "summary.json was not written"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_keys = {
        "run_id",
        "seed_id",
        "repo_id",
        "target_model",
        "mutator_model",
        "rounds_ran",
        "first_bypass_round",
        "total_bypass",
        "total_lethal",
        "total_cost_usd",
        "ablation",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["run_id"] == "summary-test"
    assert summary["rounds_ran"] == 1
    assert summary["first_bypass_round"] is None
    assert summary["total_bypass"] == 0
    assert summary["total_lethal"] == 0
    assert summary["seed_id"] == "seed-001"
    assert summary["repo_id"] == "r1"


# ---------------------------------------------------------------------------
# 7. Refusal (mutation=None) ⇒ loop continues, no sandbox call that round.
# ---------------------------------------------------------------------------


def test_driver_refusal_continues_loop(tmp_path):
    seed = _seed_single()
    repo = _repo()

    def blue_team_fn(_v, _r):
        return []

    call_state = {"n": 0}

    def agent_run_fn(parent, findings, r, **kwargs):
        call_state["n"] += 1
        if call_state["n"] == 1:
            return AgentRunResult(
                mutation=None,
                tool_chain=[],
                rationale="model refused",
                cost_usd=0.001,
                stop_reason="refusal",
                refusal=True,
            )
        return AgentRunResult(
            mutation=_mutation_of(parent, "m"),
            tool_chain=[],
            rationale="",
            cost_usd=0.002,
            stop_reason="submitted",
        )

    sandbox_mock = MagicMock(return_value=(_benign_hp(), []))

    run_dir = drv.run_evolution(
        seed,
        repo,
        rounds=2,
        blue_team="ours",
        blue_team_fn=blue_team_fn,
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_mock,
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="refusal",
    )

    rows = _read_log(run_dir / "log.jsonl")
    assert len(rows) == 2
    assert rows[0]["round"] == 1
    assert rows[0]["stop_reason"] == "refusal"
    assert rows[0]["bypassed"] is False
    assert rows[0]["lethal"] is False
    assert rows[0]["hp"] is None  # no sandbox run on refusal
    assert rows[1]["round"] == 2
    assert rows[1]["stop_reason"] == "submitted"
    # sandbox was only called for round 2.
    assert sandbox_mock.call_count == 1
