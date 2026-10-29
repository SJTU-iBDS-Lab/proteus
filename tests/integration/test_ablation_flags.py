"""Integration tests for the nine CLAW ablation flags (A1 … A9).

Each test drives the real :func:`red_team.evolution.driver.run_evolution`
with injection seams and verifies the behaviour that the specific flag is
supposed to unlock or suppress.  Coverage matrix (keep in sync with
``IMPL-22``'s ``ablation_coverage_matrix``):

* A1 — ``disable_tool='read_attack_log'``  → forwarded to ``agent_run``.
* A2 — ``disable_tool='read_skill'``       → forwarded to ``agent_run``.
* A3 — ``disable_tool='read_repo_context'``→ forwarded to ``agent_run``.
* A4 — ``tool_use_off=True``               → driver swaps in
  :class:`RandomMutation`; log rows advertise ``backend='random_baseline'``.
* A5 — ``disable_expansion=True``          → no ``is_expansion=True`` rows.
* A6 — ``search_space='evasion'``          → ``agent_run`` kwarg matches.
* A7 — ``force_topology='single'``         → sandbox sees length-1 chain.
* A8 — ``forbid_channel_edit=True``        → κ-edit mutation is rejected
  with ``stop_reason='invariant_violation_a8'`` and never sandboxed.
* A9 — ``feedback='scalar'``               → findings=[] reaches the agent.

Mocks follow the same style as ``tests/unit/evolution/test_driver_smoke.py``
so the two files can be maintained side by side.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
from sandbox.honeypot import HoneypotResult


# ---------------------------------------------------------------------------
# Shared fixtures (copied from test_driver_smoke.py for per-file independence)
# ---------------------------------------------------------------------------


def _seed_single() -> SkillVariant:
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
    """S1-valid mutation: only ``d`` changes."""
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


def _mutation_with_new_kappa(
    parent: SkillVariant, new_kappa: list[Channel]
) -> SkillVariant:
    """Mutation that also rewrites the ``kappa`` set (violates A8)."""
    metadata = dict(parent.metadata)
    metadata["mutation_tag"] = "kappa-edit"
    return SkillVariant(
        o=parent.o,
        tau=parent.tau,
        c=dict(parent.c) if parent.c else {},
        kappa=list(new_kappa),
        d=f"{parent.d}::kappa-edit",
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


def _mk_agent(parent_tag: str = "m"):
    """Factory: an ``agent_run_fn`` mock that records its kwargs per call.

    The closure's ``calls`` list captures ``(args, kwargs)`` for each call
    so tests can assert on exactly what the driver forwarded.
    """
    calls: list[tuple[tuple, dict]] = []

    def agent_run_fn(parent, findings, repo, **kwargs):
        calls.append(((parent, findings, repo), dict(kwargs)))
        return AgentRunResult(
            mutation=_mutation_of(parent, parent_tag),
            tool_chain=[],
            rationale="",
            cost_usd=0.0,
            stop_reason="submitted",
        )

    return agent_run_fn, calls


# ---------------------------------------------------------------------------
# A1 / A2 / A3: disable_tool flag flows into agent_run kwargs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,tool_name",
    [
        ("A1", "read_attack_log"),
        ("A2", "read_skill"),
        ("A3", "read_repo_context"),
    ],
)
def test_a1_a2_a3_disable_tool_forwarded_to_agent(tmp_path, label, tool_name):
    """A1/A2/A3: driver forwards ``disable_tool`` verbatim to ``agent_run``.

    The agent runtime honours this flag (see ``runtime.py`` where
    ``disabled_tool == name`` short-circuits dispatch).  For the driver's
    part we only need to prove the flag reaches the runtime; the runtime
    itself has its own unit coverage.
    """
    agent_run_fn, calls = _mk_agent(parent_tag=label.lower())

    drv.run_evolution(
        _seed_single(),
        _repo(),
        rounds=1,
        ablation={"disable_tool": tool_name},
        blue_team="ours",
        blue_team_fn=lambda _v, _r: [],
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=lambda *a, **k: (_benign_hp(), []),
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id=f"{label.lower()}-{tool_name}",
    )

    assert calls, "agent_run was never invoked"
    kwargs = calls[0][1]
    assert "run_config" in kwargs, "driver did not forward run_config kwarg"
    assert kwargs["run_config"].disabled_tool == tool_name, (
        f"{label}: expected disabled_tool={tool_name!r} in run_config, "
        f"got {kwargs['run_config'].disabled_tool!r}"
    )


def test_a1_a2_a3_agent_runtime_honours_disable_tool_contract():
    """Regression gate: the ``_RETRIEVAL_TOOLS`` set covers A1/A2/A3.

    The driver only *forwards* ``disable_tool``; the runtime is
    responsible for actually skipping dispatch.  This guard proves the
    three tool names the ablation matrix enumerates are still recognised
    as retrieval tools by :mod:`red_team.agent.runtime` — so if someone
    renames one of them the test will fail loudly and point at the right
    place.
    """
    from red_team.agent.runtime import _RETRIEVAL_TOOLS

    for name in ("read_attack_log", "read_skill", "read_repo_context"):
        assert name in _RETRIEVAL_TOOLS, (
            f"agent.runtime._RETRIEVAL_TOOLS missing {name!r}"
        )


# ---------------------------------------------------------------------------
# A4: tool_use_off ⇒ RandomMutation baseline, backend='random_baseline'.
# ---------------------------------------------------------------------------


def test_a4_tool_use_off_uses_random_baseline(tmp_path):
    agent_mock = MagicMock()

    run_dir = drv.run_evolution(
        _seed_single(),
        _repo(),
        rounds=1,
        ablation={"tool_use_off": True},
        blue_team="ours",
        blue_team_fn=lambda _v, _r: [],
        agent_run_fn=agent_mock,  # must not be called
        sandbox_run_fn=lambda *a, **k: (_benign_hp(), []),
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="a4",
    )

    rows = _read_log(run_dir / "log.jsonl")
    assert rows, "expected at least one log row"
    assert all(r["backend"] == "random_baseline" for r in rows), (
        f"expected backend=random_baseline on every row, got "
        f"{[r['backend'] for r in rows]}"
    )
    agent_mock.assert_not_called()


# ---------------------------------------------------------------------------
# A5: disable_expansion ⇒ no is_expansion=True rows.
# ---------------------------------------------------------------------------


def test_a5_disable_expansion_no_s2_s3_rows(tmp_path):
    """Force a bypass+lethal round and prove no expansion rows appear."""
    agent_run_fn, _ = _mk_agent(parent_tag="a5")
    hx_mock = MagicMock()

    run_dir = drv.run_evolution(
        _seed_single(),
        _repo(),
        rounds=1,
        ablation={"disable_expansion": True},
        blue_team="ours",
        blue_team_fn=lambda _v, _r: [],  # always bypass
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=lambda *a, **k: (_lethal_hp(), []),
        oracle_fn=lambda *a, **k: True,
        horizontal_expand_fn=hx_mock,
        output_root=tmp_path,
        run_id="a5",
    )

    rows = _read_log(run_dir / "log.jsonl")
    assert rows, "no rows emitted"
    assert all(not r["is_expansion"] for r in rows), (
        f"A5 must suppress expansion rows; got "
        f"{[r['phase'] for r in rows if r['is_expansion']]}"
    )
    hx_mock.assert_not_called()


# ---------------------------------------------------------------------------
# A6: search_space='evasion' ⇒ passed through to agent_run.
# ---------------------------------------------------------------------------


def test_a6_search_space_s1_forwarded_to_agent(tmp_path):
    """``step_s1`` hardcodes ``search_space='evasion'``; this asserts the
    driver honours the ablation's explicit override (which today also
    resolves to ``evasion``).  The test guards against a future default
    change that might silently bypass the flag.
    """
    agent_run_fn, calls = _mk_agent(parent_tag="a6")

    drv.run_evolution(
        _seed_single(),
        _repo(),
        rounds=1,
        ablation={"search_space": "evasion"},
        blue_team="ours",
        blue_team_fn=lambda _v, _r: [],
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=lambda *a, **k: (_benign_hp(), []),
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="a6",
    )

    assert calls, "agent_run was never invoked"
    kwargs = calls[0][1]
    assert kwargs.get("search_space") == "evasion", (
        f"A6: expected search_space='evasion' kwarg; got {kwargs!r}"
    )


# ---------------------------------------------------------------------------
# A7: force_topology='single' ⇒ sandbox sees chain[:1], tau=SINGLE, log row
#     carries forced_topology=True and chain_used=[first].
# ---------------------------------------------------------------------------


def test_a7_force_topology_single_truncates_chain(tmp_path):
    seed = _seed_pair()
    agent_run_fn, _ = _mk_agent(parent_tag="a7")

    sandbox_seen: list[SkillVariant] = []

    def sandbox_run_fn(variant, chain, target_model, container):
        sandbox_seen.append(variant)
        return _benign_hp(), []

    run_dir = drv.run_evolution(
        seed,
        _repo(),
        rounds=1,
        ablation={"force_topology": "single"},
        blue_team="ours",
        blue_team_fn=lambda _v, _r: [],
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_run_fn,
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="a7",
    )

    assert len(sandbox_seen) == 1
    v = sandbox_seen[0]
    assert v.tau == Topology.SINGLE
    assert len(v.chain) == 1
    assert v.chain == ["skill-alpha"]

    rows = _read_log(run_dir / "log.jsonl")
    assert rows[0]["forced_topology"] is True
    assert rows[0]["chain_original"] == ["skill-alpha", "skill-beta"]
    assert rows[0]["chain_used"] == ["skill-alpha"]


# ---------------------------------------------------------------------------
# A8: forbid_channel_edit rejects κ-edit mutations.
# ---------------------------------------------------------------------------


def test_a8_forbid_channel_edit_rejects_kappa_change(tmp_path):
    """Agent returns a mutation with a different ``kappa`` set; the driver
    must log ``stop_reason='invariant_violation_a8'`` and NOT call the
    sandbox for that round.
    """
    seed = _seed_single()
    new_kappa = [Channel.FILESYSTEM, Channel.NETWORK]  # parent has only FS
    assert set(new_kappa) != set(seed.kappa)

    def agent_run_fn(parent, findings, r, **kwargs):
        return AgentRunResult(
            mutation=_mutation_with_new_kappa(parent, new_kappa),
            tool_chain=[{"tool": "read_skill"}],
            rationale="kappa-edit attempt",
            cost_usd=0.0,
            stop_reason="submitted",
        )

    sandbox_mock = MagicMock()

    run_dir = drv.run_evolution(
        seed,
        _repo(),
        rounds=1,
        ablation={"forbid_channel_edit": True},
        blue_team="ours",
        blue_team_fn=lambda _v, _r: [],
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_mock,
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="a8",
    )

    sandbox_mock.assert_not_called()

    rows = _read_log(run_dir / "log.jsonl")
    assert len(rows) == 1
    assert rows[0]["stop_reason"] == "invariant_violation_a8"
    assert rows[0]["bypassed"] is False
    assert rows[0]["lethal"] is False
    assert rows[0]["hp"] is None


def test_a8_allows_non_kappa_mutations(tmp_path):
    """Counter-example: with A8 on, a d-only mutation must still proceed."""
    agent_run_fn, _ = _mk_agent(parent_tag="a8-ok")
    sandbox_mock = MagicMock(return_value=(_benign_hp(), []))

    drv.run_evolution(
        _seed_single(),
        _repo(),
        rounds=1,
        ablation={"forbid_channel_edit": True},
        blue_team="ours",
        blue_team_fn=lambda _v, _r: [],
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=sandbox_mock,
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="a8-ok",
    )

    sandbox_mock.assert_called_once()


# ---------------------------------------------------------------------------
# A9: feedback='scalar' hides findings from the agent.
# ---------------------------------------------------------------------------


def test_a9_feedback_scalar_passes_empty_findings(tmp_path):
    observed: list[list[PaperFinding]] = []

    def agent_run_fn(parent, findings, r, **kwargs):
        observed.append(list(findings))
        return AgentRunResult(
            mutation=_mutation_of(parent, "a9"),
            tool_chain=[],
            rationale="",
            cost_usd=0.0,
            stop_reason="submitted",
        )

    drv.run_evolution(
        _seed_single(),
        _repo(),
        rounds=1,
        ablation={"feedback": "scalar"},
        blue_team="ours",
        # Two findings would normally flow to the agent; A9 zeroes them.
        blue_team_fn=lambda _v, _r: [_finding("critical"), _finding("low")],
        agent_run_fn=agent_run_fn,
        sandbox_run_fn=lambda *a, **k: (_benign_hp(), []),
        oracle_fn=lambda *a, **k: False,
        horizontal_expand_fn=lambda *a, **k: HorizontalExpansion(),
        output_root=tmp_path,
        run_id="a9",
    )

    assert observed, "agent was never called"
    assert observed[0] == [], (
        f"A9: expected [] findings at the agent; got {observed[0]!r}"
    )
