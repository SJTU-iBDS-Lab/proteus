"""Unit tests for :mod:`red_team.evolution.horizontal_expand` (IMPL-12).

The tests mock :func:`path_expand` and :func:`surface_expand` so we only
exercise the orchestration / selection logic in ``horizontal_expand``.
Every test pins call counts and call-args via ``MagicMock`` rather than
real sandbox/LLM plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.models import RepositorySlice
from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.campaign.run_config import RunConfig
from red_team.evolution import horizontal_expand as he
from red_team.evolution.path_expansion import PathExpandResult
from red_team.evolution.surface_expansion import SurfaceFunnel, SurfaceResult
from sandbox.honeypot import HoneypotResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _parent() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=[Channel.FILESYSTEM, Channel.SUBPROCESS],
        d="seed doc",
        chain=["skill-alpha"],
        metadata={
            "motif_summary": "scheduled re-exec",
            "technique_summary": "cron write to /etc/cron.d/",
        },
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


def _mutation(parent: SkillVariant, tag: str) -> SkillVariant:
    metadata = dict(parent.metadata)
    metadata["tag"] = tag
    return SkillVariant(
        o=parent.o,
        tau=parent.tau,
        c=dict(parent.c),
        kappa=list(parent.kappa),
        d=f"{parent.d}::{tag}",
        poison_index=parent.poison_index,
        chain=list(parent.chain),
        metadata=metadata,
    )


def _path_result(
    parent: SkillVariant,
    *,
    tag: str,
    bypassed: bool,
    lethal: bool,
    score: float,
) -> PathExpandResult:
    return PathExpandResult(
        mutation=_mutation(parent, tag),
        tool_chain=[{"tool": "read_skill"}],
        hp=HoneypotResult(files_created=[f"/tmp/{tag}"]),
        bypassed=bypassed,
        lethal=lethal,
        family_candidate={"name": tag},
        score=score,
        rationale=f"{tag} rationale",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_horizontal_expand_disable_returns_empty(monkeypatch):
    """run_config.expansion_enabled=False short-circuits both sub-calls."""
    parent = _parent()
    path_stub = MagicMock()
    surface_stub = MagicMock()

    monkeypatch.setattr(he, "path_expand", path_stub)
    monkeypatch.setattr(he, "surface_expand", surface_stub)

    rc = RunConfig.from_ablation({"disable_expansion": True})
    result = he.horizontal_expand(
        parent,
        _repo(),
        agent_run=MagicMock(),
        run_in_sandbox=MagicMock(),
        oracle_lethal=MagicMock(),
        attack_log_path=None,
        run_config=rc,
    )

    assert isinstance(result, he.HorizontalExpansion)
    assert result.path_results == []
    assert isinstance(result.surface_funnel, SurfaceFunnel)
    assert result.surface_funnel.ideated == []
    assert result.surface_funnel.selected == []
    assert result.surface_funnel.executed == []
    assert result.surface_funnel.succeeded == 0
    path_stub.assert_not_called()
    surface_stub.assert_not_called()


def test_horizontal_expand_picks_best_path_result(monkeypatch):
    """Highest-score bypassed+lethal row is the S3 seed."""
    parent = _parent()
    rows = [
        _path_result(parent, tag="r1", bypassed=False, lethal=False, score=1.0),
        _path_result(parent, tag="r2", bypassed=True, lethal=True, score=5.0),
        _path_result(parent, tag="r3", bypassed=True, lethal=True, score=3.0),
    ]
    returned_funnel = SurfaceFunnel(
        ideated=[{"o_prime": "data_exfiltration"}],
        selected=[{"o_prime": "data_exfiltration"}],
        executed=[],
        succeeded=0,
    )

    path_stub = MagicMock(return_value=rows)
    surface_stub = MagicMock(return_value=returned_funnel)
    monkeypatch.setattr(he, "path_expand", path_stub)
    monkeypatch.setattr(he, "surface_expand", surface_stub)

    agent_run = MagicMock()
    sandbox = MagicMock()
    oracle = MagicMock()

    result = he.horizontal_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
    )

    # path_expand invoked exactly once with parent.
    assert path_stub.call_count == 1
    assert path_stub.call_args.args[0] is parent

    # surface_expand invoked with the score=5.0 mutation (tag r2).
    assert surface_stub.call_count == 1
    seed_passed = surface_stub.call_args.args[0]
    assert seed_passed is rows[1].mutation
    assert seed_passed.metadata["tag"] == "r2"

    # Result wires through.
    assert result.path_results == rows
    assert result.surface_funnel is returned_funnel


def test_horizontal_expand_falls_back_to_parent_when_no_success(monkeypatch):
    """No row bypassed+lethal ⇒ parent is passed to surface_expand."""
    parent = _parent()
    rows = [
        _path_result(parent, tag="r1", bypassed=False, lethal=True, score=4.0),
        _path_result(parent, tag="r2", bypassed=True, lethal=False, score=7.0),
    ]

    path_stub = MagicMock(return_value=rows)
    surface_stub = MagicMock(return_value=SurfaceFunnel())
    monkeypatch.setattr(he, "path_expand", path_stub)
    monkeypatch.setattr(he, "surface_expand", surface_stub)

    result = he.horizontal_expand(
        parent,
        _repo(),
        agent_run=MagicMock(),
        run_in_sandbox=MagicMock(),
        oracle_lethal=MagicMock(),
        attack_log_path=None,
    )

    assert surface_stub.call_count == 1
    assert surface_stub.call_args.args[0] is parent
    assert result.path_results == rows


def test_horizontal_expand_empty_path_results_uses_parent(monkeypatch):
    """path_expand returns [] ⇒ surface_expand called with parent."""
    parent = _parent()

    path_stub = MagicMock(return_value=[])
    surface_stub = MagicMock(return_value=SurfaceFunnel())
    monkeypatch.setattr(he, "path_expand", path_stub)
    monkeypatch.setattr(he, "surface_expand", surface_stub)

    result = he.horizontal_expand(
        parent,
        _repo(),
        agent_run=MagicMock(),
        run_in_sandbox=MagicMock(),
        oracle_lethal=MagicMock(),
        attack_log_path=None,
    )

    assert path_stub.call_count == 1
    assert surface_stub.call_count == 1
    assert surface_stub.call_args.args[0] is parent
    assert result.path_results == []


def test_horizontal_expand_propagates_surface_funnel(monkeypatch):
    """Returned SurfaceFunnel is whatever surface_expand produced."""
    parent = _parent()
    mutation = _mutation(parent, "s3")
    executed_row = SurfaceResult(
        mutation=mutation,
        hp=HoneypotResult(files_created=["/tmp/s3"]),
        bypassed=True,
        lethal=True,
        o_prime=AttackObjective.PERSISTENCE,
        shared_motif={"kappa": ["filesystem"]},
        post_hoc_ok=True,
        rationale="transfer",
    )
    funnel = SurfaceFunnel(
        ideated=[{"o_prime": "data_exfiltration"}],
        selected=[{"o_prime": "data_exfiltration"}],
        executed=[executed_row],
        succeeded=1,
    )

    path_stub = MagicMock(return_value=[])
    surface_stub = MagicMock(return_value=funnel)
    monkeypatch.setattr(he, "path_expand", path_stub)
    monkeypatch.setattr(he, "surface_expand", surface_stub)

    result = he.horizontal_expand(
        parent,
        _repo(),
        agent_run=MagicMock(),
        run_in_sandbox=MagicMock(),
        oracle_lethal=MagicMock(),
        attack_log_path=None,
    )

    assert result.surface_funnel is funnel
    assert result.surface_funnel.succeeded == 1
    assert result.surface_funnel.executed == [executed_row]
