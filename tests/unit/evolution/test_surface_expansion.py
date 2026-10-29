"""Unit tests for :mod:`red_team.evolution.surface_expansion` (IMPL-11).

Covers the ideation \u2192 selection \u2192 execution funnel plus the strict
:func:`post_hoc_shared_motif` filter.  Every LLM call is mocked through a
monkey-patched ``_make_brainstorm_client`` (same pattern as
``test_family_enumeration``); ``agent_run`` / ``run_in_sandbox`` /
``oracle_lethal`` are plain stubs so we can assert call counts and pin
outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.baselines.base import AgentRunResult
from red_team.campaign.run_config import RunConfig
from red_team.evolution import surface_expansion as se
from sandbox.honeypot import HoneypotResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _parent(
    *,
    kappa: list[Channel] | None = None,
    tool_chain_prefix: list[str] | None = None,
    file_access: list[str] | None = None,
    entry_script: str = "run.sh",
) -> SkillVariant:
    metadata: dict[str, object] = {"motif_summary": "scheduled re-exec"}
    if tool_chain_prefix is not None:
        metadata["tool_chain_prefix"] = list(tool_chain_prefix)
    if file_access is not None:
        metadata["file_access"] = list(file_access)
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={"entry_script": entry_script},
        kappa=kappa if kappa is not None else [Channel.FILESYSTEM, Channel.SUBPROCESS],
        d="seed doc",
        chain=["skill-alpha"],
        metadata=metadata,
    )


def _child(
    parent: SkillVariant,
    *,
    o: AttackObjective = AttackObjective.DATA_EXFILTRATION,
    kappa: list[Channel] | None = None,
    tool_chain_prefix: list[str] | None = None,
    file_access: list[str] | None = None,
    entry_script: str | None = None,
) -> SkillVariant:
    metadata = dict(parent.metadata)
    if tool_chain_prefix is not None:
        metadata["tool_chain_prefix"] = list(tool_chain_prefix)
    else:
        metadata.pop("tool_chain_prefix", None)
    if file_access is not None:
        metadata["file_access"] = list(file_access)
    else:
        metadata.pop("file_access", None)
    c = dict(parent.c)
    if entry_script is not None:
        c["entry_script"] = entry_script
    return SkillVariant(
        o=o,
        tau=parent.tau,
        c=c,
        kappa=kappa if kappa is not None else list(parent.kappa),
        d=parent.d,
        poison_index=parent.poison_index,
        chain=list(parent.chain),
        metadata=metadata,
    )


def _make_llm_response(payload: object) -> SimpleNamespace:
    """Build a minimal object that looks like an OpenAI chat completion."""
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _StubClient:
    """Stand-in for ``openai.OpenAI`` used by the ideation/selection calls."""

    def __init__(self, responses: list[object]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=self.completions)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("stub client exhausted")
        payload = self._responses.pop(0)
        return _make_llm_response(payload)


@pytest.fixture
def patch_client(monkeypatch):
    """Return a helper that monkey-patches ``_make_brainstorm_client``."""

    def _apply(responses: list[object]) -> _StubClient:
        stub = _StubClient(responses)
        monkeypatch.setattr(
            se, "_make_brainstorm_client", lambda base_url, api_key: stub
        )
        return stub

    return _apply


# ---------------------------------------------------------------------------
# Candidate + agent_run helpers
# ---------------------------------------------------------------------------


def _ideation_candidate(
    o_prime: str,
    *,
    kappa: list[str] | None = None,
    tool_chain_prefix: list[str] | None = None,
    file_access: list[str] | None = None,
    feasibility: str = "",
) -> dict[str, object]:
    return {
        "o_prime": o_prime,
        "shared_motif": {
            "kappa": list(kappa or ["filesystem"]),
            "tool_chain_prefix": list(tool_chain_prefix or []),
            "file_access": list(file_access or []),
        },
        "feasibility": feasibility or f"transfer-to-{o_prime}",
    }


def _make_agent_run_returning(mutations: list[SkillVariant | None]) -> MagicMock:
    """Return an ``agent_run`` stub that yields mutations in order."""
    queue = list(mutations)

    def _stub(seed, findings, repo, *, search_space, **kwargs):
        assert search_space == "surface_expansion"
        assert kwargs.get("round_index") == -1
        if not queue:
            raise AssertionError("agent_run stub exhausted")
        mutation = queue.pop(0)
        return AgentRunResult(
            mutation=mutation,
            tool_chain=[{"tool": "read_skill"}],
            rationale="ok",
            cost_usd=0.0,
            stop_reason="submitted" if mutation is not None else "refusal",
        )

    return MagicMock(side_effect=_stub)


def _sandbox_stub() -> MagicMock:
    def _stub(variant, chain, target_model, container):
        return HoneypotResult(files_created=["/tmp/x"]), []

    return MagicMock(side_effect=_stub)


# ---------------------------------------------------------------------------
# Ideation / selection / execution funnel
# ---------------------------------------------------------------------------


def test_surface_expand_full_funnel(patch_client):
    """5 ideated \u2192 3 selected \u2192 3 executed \u2192 2 pass all three gates."""
    parent = _parent(
        tool_chain_prefix=["read_file", "run_subprocess"],
        file_access=["/etc/cron.d"],
    )

    ideated = [
        _ideation_candidate("data_exfiltration"),
        _ideation_candidate("privilege_escalation"),
        _ideation_candidate("cleanup"),
        _ideation_candidate("guidance_injection"),
        _ideation_candidate("supply_chain_abuse"),
    ]
    # Selection picks indices [0, 1, 2] \u2192 first three candidates.
    patch_client([ideated, [0, 1, 2]])

    # Three child mutations, all sharing \u03ba + tool_chain_prefix (post_hoc ok).
    child_0 = _child(
        parent,
        o=AttackObjective.DATA_EXFILTRATION,
        tool_chain_prefix=["read_file", "run_subprocess"],
        file_access=["/etc/cron.d"],
    )
    child_1 = _child(
        parent,
        o=AttackObjective.PRIVILEGE_ESCALATION,
        tool_chain_prefix=["read_file", "run_subprocess"],
        file_access=["/etc/cron.d"],
    )
    child_2 = _child(
        parent,
        o=AttackObjective.CLEANUP,
        tool_chain_prefix=["read_file", "run_subprocess"],
        file_access=["/etc/cron.d"],
    )
    agent_run = _make_agent_run_returning([child_0, child_1, child_2])
    sandbox = _sandbox_stub()

    # Oracle: first two lethal, third not.
    verdicts = iter([True, True, False])
    oracle = MagicMock(side_effect=lambda *a, **kw: next(verdicts))

    funnel = se.surface_expand(
        parent,
        repo=SimpleNamespace(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        k=5,
        k_prime=3,
    )

    assert len(funnel.ideated) == 5
    assert len(funnel.selected) == 3
    assert len(funnel.executed) == 3
    # ``funnel.succeeded`` is now driver-owned (Q4-A): surface_expand
    # itself leaves it at 0 and the campaign runner overwrites it after
    # blue-team re-audit.  Here we assert the *evolution-layer* signal
    # (lethal AND post_hoc_ok) which is what the driver will combine
    # with its bypass verdict — 2 of the 3 mutations satisfy it.
    assert funnel.succeeded == 0
    assert sum(
        1 for r in funnel.executed if r.lethal and r.post_hoc_ok
    ) == 2
    # All three executed results recorded post_hoc_ok True (same motif).
    assert all(r.post_hoc_ok for r in funnel.executed)
    # The third one failed the oracle so it isn't counted even though post_hoc ok.
    assert funnel.executed[2].lethal is False
    # Selection used indices 0..2.
    assert [r.o_prime for r in funnel.executed] == [
        AttackObjective.DATA_EXFILTRATION,
        AttackObjective.PRIVILEGE_ESCALATION,
        AttackObjective.CLEANUP,
    ]
    # Sandbox invoked once per executed mutation.
    assert sandbox.call_count == 3


def test_surface_expand_post_hoc_rejects_motif_drift(patch_client):
    """Oracle=True but post_hoc fails \u2192 result recorded, succeeded unchanged."""
    parent = _parent(
        tool_chain_prefix=["read_file"],
        file_access=["/etc/cron.d"],
    )
    ideated = [_ideation_candidate("data_exfiltration")]
    patch_client([ideated, [0]])

    # Child shares only \u03ba, nothing else \u2014 post_hoc needs 2 axes, fails.
    drifted = _child(
        parent,
        o=AttackObjective.DATA_EXFILTRATION,
        kappa=[Channel.FILESYSTEM],  # overlaps parent \u03ba
        tool_chain_prefix=["totally_different_tool"],
        file_access=["/var/log/unrelated"],
    )
    agent_run = _make_agent_run_returning([drifted])
    oracle = MagicMock(return_value=True)

    funnel = se.surface_expand(
        parent,
        repo=SimpleNamespace(),
        agent_run=agent_run,
        run_in_sandbox=_sandbox_stub(),
        oracle_lethal=oracle,
        attack_log_path=None,
        k=1,
        k_prime=1,
    )

    assert len(funnel.executed) == 1
    assert funnel.executed[0].lethal is True
    assert funnel.executed[0].post_hoc_ok is False
    # Q4-A: succeeded is driver-owned; here the trivial 0 still holds
    # because the evolution-layer signal (lethal AND post_hoc_ok) is also 0.
    assert funnel.succeeded == 0
    assert funnel.executed[0].bypassed is None


def test_surface_expand_json_parse_fail(patch_client):
    """Three malformed replies \u2192 empty funnel, no agent_run spent."""
    stub = patch_client(["garbage", "{nope", "<html>oops</html>"])
    parent = _parent()
    agent_run = MagicMock()

    funnel = se.surface_expand(
        parent,
        repo=SimpleNamespace(),
        agent_run=agent_run,
        run_in_sandbox=_sandbox_stub(),
        oracle_lethal=MagicMock(return_value=True),
        attack_log_path=None,
        k=5,
        k_prime=2,
    )

    assert funnel.ideated == []
    assert funnel.selected == []
    assert funnel.executed == []
    assert funnel.succeeded == 0
    # All three ideation attempts were actually issued (no selection yet).
    assert len(stub.calls) == 3
    agent_run.assert_not_called()


def test_surface_expand_ablation_disable():
    """``disable_expansion`` short-circuits before any LLM call."""
    parent = _parent()
    agent_run = MagicMock()
    sandbox = MagicMock()

    # No patch_client \u2014 if any LLM call slipped through, the import would fail.
    funnel = se.surface_expand(
        parent,
        repo=SimpleNamespace(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=MagicMock(),
        attack_log_path=None,
        run_config=RunConfig.from_ablation({"disable_expansion": True}),
    )

    assert funnel.ideated == []
    assert funnel.selected == []
    assert funnel.executed == []
    assert funnel.succeeded == 0
    agent_run.assert_not_called()
    sandbox.assert_not_called()


def test_surface_expand_selection_fallback(patch_client):
    """Selection LLM garbage \u2192 fallback to first k_prime indices."""
    parent = _parent(
        tool_chain_prefix=["read_file"],
        file_access=["/etc/cron.d"],
    )
    ideated = [
        _ideation_candidate("data_exfiltration"),
        _ideation_candidate("privilege_escalation"),
        _ideation_candidate("cleanup"),
        _ideation_candidate("guidance_injection"),
    ]
    # Ideation OK (first call), then three garbage replies for selection.
    patch_client([ideated, "nope", "{still_garbage", "<html/>"])

    child_a = _child(
        parent,
        o=AttackObjective.DATA_EXFILTRATION,
        tool_chain_prefix=["read_file"],
        file_access=["/etc/cron.d"],
    )
    child_b = _child(
        parent,
        o=AttackObjective.PRIVILEGE_ESCALATION,
        tool_chain_prefix=["read_file"],
        file_access=["/etc/cron.d"],
    )
    agent_run = _make_agent_run_returning([child_a, child_b])

    funnel = se.surface_expand(
        parent,
        repo=SimpleNamespace(),
        agent_run=agent_run,
        run_in_sandbox=_sandbox_stub(),
        oracle_lethal=MagicMock(return_value=True),
        attack_log_path=None,
        k=4,
        k_prime=2,
    )

    # Fallback picks first 2 candidates \u2192 data_exfiltration, privilege_escalation.
    assert len(funnel.selected) == 2
    assert [c["o_prime"] for c in funnel.selected] == [
        "data_exfiltration",
        "privilege_escalation",
    ]
    assert len(funnel.executed) == 2
    # Q4-A: ``funnel.succeeded`` is driver-owned; assert evolution-layer
    # signal directly (both rows satisfy lethal AND post_hoc_ok).
    assert funnel.succeeded == 0
    assert sum(
        1 for r in funnel.executed if r.lethal and r.post_hoc_ok
    ) == 2


def test_surface_expand_skips_same_objective_candidate(patch_client):
    """``o_prime`` equal to parent.o is dropped silently."""
    parent = _parent()
    ideated = [
        _ideation_candidate("persistence"),  # same as parent \u2014 must drop
        _ideation_candidate("data_exfiltration"),
    ]
    patch_client([ideated, [0, 1]])

    child = _child(
        parent,
        o=AttackObjective.DATA_EXFILTRATION,
        tool_chain_prefix=["read_file"],
        file_access=["/etc/cron.d"],
    )
    agent_run = _make_agent_run_returning([child])

    # Need post_hoc pass: give parent matching motif axes.
    parent = _parent(
        tool_chain_prefix=["read_file"],
        file_access=["/etc/cron.d"],
    )

    funnel = se.surface_expand(
        parent,
        repo=SimpleNamespace(),
        agent_run=agent_run,
        run_in_sandbox=_sandbox_stub(),
        oracle_lethal=MagicMock(return_value=True),
        attack_log_path=None,
        k=2,
        k_prime=2,
    )

    # Only one executed (the non-persistence candidate).
    assert len(funnel.selected) == 2
    assert len(funnel.executed) == 1
    assert funnel.executed[0].o_prime == AttackObjective.DATA_EXFILTRATION


# ---------------------------------------------------------------------------
# post_hoc_shared_motif direct tests
# ---------------------------------------------------------------------------


def test_post_hoc_shared_motif_two_axes_pass():
    """\u03ba overlap + tool_chain_prefix overlap \u2192 True."""
    parent = _parent(
        tool_chain_prefix=["read_file", "run_subprocess"],
    )
    child = _child(
        parent,
        o=AttackObjective.DATA_EXFILTRATION,
        tool_chain_prefix=["read_file", "curl_post"],
    )
    assert se.post_hoc_shared_motif(parent, child) is True


def test_post_hoc_shared_motif_one_axis_fails():
    """Only \u03ba overlap, no tool_chain/file_access \u2192 False (strict)."""
    parent = _parent(tool_chain_prefix=None, file_access=None)
    child = _child(parent, o=AttackObjective.DATA_EXFILTRATION)
    # Parent + child share \u03ba, nothing else.
    assert se.post_hoc_shared_motif(parent, child) is False


def test_post_hoc_shared_motif_entry_script_no_fallback():
    """Same ``entry_script`` alone is NOT a motif axis in the strict variant."""
    parent = SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=[Channel.FILESYSTEM],
        d="parent doc",
        chain=["skill-alpha"],
        metadata={},
    )
    child = SkillVariant(
        o=AttackObjective.DATA_EXFILTRATION,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},  # same entry_script
        kappa=[Channel.NETWORK],        # no \u03ba overlap
        d="child doc",
        chain=["skill-alpha"],
        metadata={},
    )
    # Shared entry_script but no \u03ba/prefix/files axis \u2192 False.
    assert se.post_hoc_shared_motif(parent, child) is False


def test_post_hoc_shared_motif_three_axes_pass():
    """\u03ba + tool_chain_prefix + file_access all overlap \u2192 True."""
    parent = _parent(
        tool_chain_prefix=["read_file"],
        file_access=["/etc/cron.d", "/tmp/scratch"],
    )
    child = _child(
        parent,
        o=AttackObjective.DATA_EXFILTRATION,
        tool_chain_prefix=["read_file", "curl"],
        file_access=["/etc/cron.d"],
    )
    assert se.post_hoc_shared_motif(parent, child) is True


def test_post_hoc_shared_motif_file_access_only_no_kappa():
    """No \u03ba overlap + only file_access \u2192 single axis \u2192 False."""
    parent = _parent(
        kappa=[Channel.FILESYSTEM],
        file_access=["/etc/cron.d"],
    )
    child = _child(
        parent,
        o=AttackObjective.DATA_EXFILTRATION,
        kappa=[Channel.NETWORK],  # no overlap with filesystem
        file_access=["/etc/cron.d"],
    )
    assert se.post_hoc_shared_motif(parent, child) is False
