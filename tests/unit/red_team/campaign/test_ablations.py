"""Unit tests for :mod:`red_team.campaign.ablations` hooks.

One test per hook so a regression points at a single file.  The
integration coverage (``tests/integration/test_ablation_flags.py``) then
proves the runner actually calls the hooks at the right points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import (
    AttackObjective,
    Channel,
    Evidence,
    PaperFinding,
    SkillVariant,
    Topology,
)
from red_team.campaign import ablations as ablations_pkg
from red_team.campaign.ablations import (
    disable_expansion,
    disable_tool,
    forbid_channel_edit,
    force_topology_single,
    scalar_feedback,
    search_space,
    tool_use_off,
)
from red_team.campaign.run_config import RunConfig
from red_team.evolution.search_space import InvariantViolation


def _variant(
    chain=None, kappa=None, tau: Topology = Topology.SINGLE
) -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=tau,
        c={"entry_script": "run.sh"},
        kappa=list(kappa or [Channel.FILESYSTEM]),
        d="doc",
        chain=list(chain or ["skill-alpha"]),
        metadata={"seed_id": "s"},
    )


def _finding(sev: str = "low") -> PaperFinding:
    return PaperFinding(
        finding_id="x",
        layer="A",
        category="c",
        reason_code="r",
        confidence=0.9,
        severity=sev,
        evidence=Evidence(snippet=""),
        explanation="e",
    )


# --- A9 ---------------------------------------------------------------


def test_a9_scalar_feedback_zeroes_findings():
    rc = RunConfig.from_ablation({"feedback": "scalar"})
    out = scalar_feedback.filter_findings([_finding("critical")], rc)
    assert out == []


def test_a9_off_passes_findings_through():
    fs = [_finding("low")]
    rc = RunConfig.from_ablation({})
    assert scalar_feedback.filter_findings(fs, rc) == fs


# --- A4 ---------------------------------------------------------------


def test_a4_tool_use_off_triggers_short_circuit():
    rc = RunConfig.from_ablation({"tool_use_off": True})
    out = tool_use_off.try_short_circuit_agent(rc)
    assert out == (True, "random_baseline")


def test_a4_off_returns_none():
    rc = RunConfig.from_ablation({})
    assert tool_use_off.try_short_circuit_agent(rc) is None


# --- A7 ---------------------------------------------------------------


def test_a7_force_single_truncates_pair():
    v = _variant(
        chain=["a", "b"],
        kappa=[Channel.FILESYSTEM, Channel.SUBPROCESS],
        tau=Topology.PAIR,
    )
    rc = RunConfig.from_ablation({"force_topology": "single"})
    forced, flag, original, used = force_topology_single.maybe_force_single_topology(v, rc)
    assert flag is True
    assert original == ["a", "b"]
    assert used == ["a"]
    assert forced.tau == Topology.SINGLE
    assert forced.chain == ["a"]


def test_a7_off_returns_variant_untouched():
    v = _variant(chain=["a"])
    rc = RunConfig.from_ablation({})
    forced, flag, original, used = force_topology_single.maybe_force_single_topology(v, rc)
    assert flag is False and forced is v and original is None and used is None


def test_a7_already_single_passes_through():
    v = _variant(chain=["a"], tau=Topology.SINGLE)
    rc = RunConfig.from_ablation({"force_topology": "single"})
    forced, flag, original, used = force_topology_single.maybe_force_single_topology(v, rc)
    assert flag is True and original == ["a"] and used == ["a"]


# --- A5 ---------------------------------------------------------------


def test_a5_expansion_disabled_when_flag_set():
    rc_off = RunConfig.from_ablation({"disable_expansion": True})
    rc_on = RunConfig.from_ablation({})
    assert disable_expansion.is_expansion_disabled(rc_off)
    assert not disable_expansion.is_expansion_disabled(rc_on)


# --- A8 ---------------------------------------------------------------


def test_a8_kappa_edit_raises():
    parent = _variant(kappa=[Channel.FILESYSTEM])
    child = _variant(kappa=[Channel.FILESYSTEM, Channel.NETWORK])
    rc = RunConfig.from_ablation({"forbid_channel_edit": True})
    with pytest.raises(InvariantViolation):
        forbid_channel_edit.check_channel_invariant(parent, child, rc)


def test_a8_equal_kappa_passes():
    parent = _variant(kappa=[Channel.FILESYSTEM])
    child = _variant(kappa=[Channel.FILESYSTEM])
    rc = RunConfig.from_ablation({"forbid_channel_edit": True})
    forbid_channel_edit.check_channel_invariant(parent, child, rc)


def test_a8_flag_off_allows_kappa_edit():
    parent = _variant(kappa=[Channel.FILESYSTEM])
    child = _variant(kappa=[Channel.NETWORK])
    rc = RunConfig.from_ablation({})
    forbid_channel_edit.check_channel_invariant(parent, child, rc)


# --- A1/A2/A3 / A6 documentation-only hooks ---------------------------


def test_a1_a2_a3_recognised_tools():
    assert set(disable_tool.RECOGNISED_TOOLS) == {
        "read_attack_log", "read_skill", "read_repo_context"
    }


def test_a6_resolve_search_space():
    rc_default = RunConfig.from_ablation({})
    # default search_space is "free_mutation" in RunConfig; test the module
    assert search_space.resolve_search_space(rc_default) == "free_mutation"
    rc_path = RunConfig.from_ablation({"search_space": "path_expansion"})
    assert search_space.resolve_search_space(rc_path) == "path_expansion"


# --- registry composition --------------------------------------------


def test_apply_feedback_hooks_runs_scalar_filter():
    fs = [_finding()]
    rc = RunConfig.from_ablation({"feedback": "scalar"})
    assert ablations_pkg.apply_feedback_hooks(fs, rc) == []


def test_select_agent_dispatch_returns_baseline_tuple():
    rc = RunConfig.from_ablation({"tool_use_off": True})
    out = ablations_pkg.select_agent_dispatch(rc)
    assert out == (True, "random_baseline")


def test_check_mutation_invariants_raises_on_kappa_edit():
    parent = _variant(kappa=[Channel.FILESYSTEM])
    child = _variant(kappa=[Channel.NETWORK])
    rc = RunConfig.from_ablation({"forbid_channel_edit": True})
    with pytest.raises(InvariantViolation):
        ablations_pkg.check_mutation_invariants(parent, child, rc)


def test_apply_sandbox_rewriters_noop_when_unflagged():
    v = _variant()
    rc = RunConfig.from_ablation({})
    assert ablations_pkg.apply_sandbox_rewriters(v, rc) == (v, False, None, None)


def test_is_expansion_enabled_respects_flag():
    rc_on = RunConfig.from_ablation({})
    rc_off = RunConfig.from_ablation({"disable_expansion": True})
    assert ablations_pkg.is_expansion_enabled(rc_on) is True
    assert ablations_pkg.is_expansion_enabled(rc_off) is False
