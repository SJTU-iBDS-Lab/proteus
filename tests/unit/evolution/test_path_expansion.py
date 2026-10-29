"""Unit tests for :mod:`red_team.evolution.path_expansion` (IMPL-10).

The tests exercise three orthogonal axes:

* The family-enumerate call is mocked to return a controlled set of
  candidates (or ``[]``).
* ``agent_run`` is a stub that emits an :class:`AgentRunResult` with a
  fresh :class:`SkillVariant`.
* ``run_in_sandbox`` + ``oracle_lethal`` are plain stubs so we can
  count invocations and pin outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.models import RepositorySlice
from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.baselines.base import AgentRunResult
from red_team.campaign.run_config import RunConfig
from red_team.evolution import path_expansion as pe
from red_team.evolution.family_enumeration import FamilyCandidate
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


def _candidate(name: str) -> FamilyCandidate:
    return FamilyCandidate(
        name=name,
        mechanism=f"{name} mechanism",
        kappa=["filesystem"],
        requires=[],
        benign_use_keywords=[name],
        known_canary_overlap="",
        benign_cover=3,
        history=[],
        kappa_fit=True,
        score=3.0,
    )


def _make_mutation(parent: SkillVariant, tag: str) -> SkillVariant:
    # SkillVariant has slots=True + immutable-ish fields; build fresh.
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


def _make_agent_run(parent: SkillVariant) -> MagicMock:
    """Build an ``agent_run`` stub that emits a tagged mutation each call."""
    counter = {"n": 0}

    def _stub(seed, findings, repo, *, search_space, **kwargs):
        counter["n"] += 1
        mutation = _make_mutation(seed, f"r{counter['n']}")
        return AgentRunResult(
            mutation=mutation,
            tool_chain=[{"tool": "read_skill"}],
            rationale="ok",
            cost_usd=0.0,
            stop_reason="submitted",
        )

    return MagicMock(side_effect=_stub)


def _make_sandbox() -> MagicMock:
    """``run_in_sandbox`` stub that always returns a benign HoneypotResult."""

    def _stub(variant, chain, target_model, container):
        return HoneypotResult(files_created=["/etc/cron.d/wat"]), []

    return MagicMock(side_effect=_stub)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_path_expand_single_round_exhaustive_runs_all_top_3(monkeypatch):
    """C9 exhaustive flag: ``family_enumerate`` returns 3 ⇒ all 3 are run.

    Pre-C9 this was the default behaviour; under C9 the default is
    early-stop on first lethal hit.  This test pins the
    ``ablation['path_expand_exhaustive']=True`` path which keeps the
    pre-C9 "every top-k candidate" semantics for the §"family-enum
    recall" ablation column.
    """
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    monkeypatch.setattr(
        pe,
        "family_enumerate",
        lambda *a, **kw: [
            _candidate("cron"),
            _candidate("systemd"),
            _candidate("bashrc"),
        ],
    )

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=3,
        run_config=RunConfig.from_ablation({"path_expand_exhaustive": True}),
    )

    assert len(out) == 3
    # agent_run called exactly 3 times.
    assert agent_run.call_count == 3
    # sandbox called 3 times too.
    assert sandbox.call_count == 3
    # Each result carries its candidate dict.
    names = [r.family_candidate["name"] for r in out]
    assert set(names) == {"cron", "systemd", "bashrc"}
    assert all(r.lethal for r in out)


def test_path_expand_on_result_hook_fires_per_entry(monkeypatch):
    """``on_result`` must fire once per appended PathExpandResult."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    monkeypatch.setattr(
        pe,
        "family_enumerate",
        lambda *a, **kw: [
            _candidate("cron"),
            _candidate("systemd"),
            _candidate("bashrc"),
        ],
    )

    received: list = []
    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=3,
        run_config=RunConfig.from_ablation({"path_expand_exhaustive": True}),
        on_result=lambda r: received.append(r),
    )

    assert len(out) == 3
    assert len(received) == 3
    # Hook was called with the same PathExpandResult instances appended.
    assert received == out


def test_path_expand_on_result_hook_swallows_exceptions(monkeypatch):
    """A faulty ``on_result`` must not abort the inner loop."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)
    monkeypatch.setattr(
        pe, "family_enumerate",
        lambda *a, **kw: [_candidate("cron"), _candidate("systemd")],
    )

    def boom(_):
        raise RuntimeError("hook crash")

    out = pe.path_expand(
        parent, _repo(),
        agent_run=agent_run, run_in_sandbox=sandbox, oracle_lethal=oracle,
        attack_log_path=None, max_rounds=1, top_k=2,
        run_config=RunConfig.from_ablation({"path_expand_exhaustive": True}),
        on_result=boom,
    )
    # Both candidates still produced results despite hook crash.
    assert len(out) == 2


def test_path_expand_ablation_disable_family_enumeration(monkeypatch):
    """Ablation collapses S2 to a direct agent_run per round, no family call."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    fe_stub = MagicMock()
    monkeypatch.setattr(pe, "family_enumerate", fe_stub)

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=3,
        run_config=RunConfig.from_ablation({"disable_family_enumeration": True}),
    )

    assert len(out) == 1
    assert fe_stub.call_count == 0
    assert agent_run.call_count == 1
    assert out[0].family_candidate is None


def test_path_expand_max_rounds_zero_candidates(monkeypatch):
    """family_enumerate returns []: path_expand bails immediately."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    monkeypatch.setattr(pe, "family_enumerate", lambda *a, **kw: [])

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=3,
        top_k=3,
    )
    assert out == []
    assert agent_run.call_count == 0
    assert sandbox.call_count == 0


def test_path_expand_agent_refusal_skips(monkeypatch):
    """If agent_run returns mutation=None we skip without appending."""
    parent = _parent()
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    def _refuse(seed, findings, repo, *, search_space, **kwargs):
        return AgentRunResult(mutation=None, stop_reason="refusal")

    agent_run = MagicMock(side_effect=_refuse)

    monkeypatch.setattr(
        pe,
        "family_enumerate",
        lambda *a, **kw: [_candidate("cron"), _candidate("systemd")],
    )

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=2,
    )
    assert out == []
    # Sandbox is never engaged because agent produced no mutation.
    assert sandbox.call_count == 0


def test_path_expand_multi_round_updates_parent(monkeypatch):
    """After a lethal bypass, the next round should brainstorm from that variant."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    seen_parents: list[SkillVariant] = []

    def _fe_stub(current_parent, *args, **kwargs):
        seen_parents.append(current_parent)
        return [_candidate("cron")]

    monkeypatch.setattr(pe, "family_enumerate", _fe_stub)

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=2,
        top_k=1,
    )

    assert len(out) == 2
    # First round sees the original parent; second round sees the evolved one.
    assert seen_parents[0] is parent
    assert seen_parents[1] is not parent
    # The evolved parent is the mutation produced in round 1.
    assert seen_parents[1].d.endswith("::r1")


def test_path_expand_family_hint_flows_to_agent(monkeypatch):
    """The candidate should be wired into both metadata and ablation."""
    parent = _parent()
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    seen_calls: list[dict] = []

    def _agent(seed, findings, repo, *, search_space, **kwargs):
        rc = kwargs.get("run_config")
        family_hint_rc = rc.family_hint if rc is not None else None
        seen_calls.append({
            "family_hint_meta": seed.metadata.get("family_hint"),
            "family_hint_rc": family_hint_rc,
            "search_space": search_space,
        })
        return AgentRunResult(
            mutation=_make_mutation(seed, "r1"),
            tool_chain=[],
            stop_reason="submitted",
        )

    monkeypatch.setattr(
        pe, "family_enumerate", lambda *a, **kw: [_candidate("cron")]
    )

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=_agent,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=1,
    )

    assert len(out) == 1
    assert len(seen_calls) == 1
    call = seen_calls[0]
    assert call["search_space"] == "path_expansion"
    assert call["family_hint_meta"]["name"] == "cron"
    assert call["family_hint_rc"]["name"] == "cron"


def test_path_expand_result_to_dict_roundtrip(monkeypatch):
    """``PathExpandResult.to_dict`` produces a JSON-safe payload."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    monkeypatch.setattr(
        pe, "family_enumerate", lambda *a, **kw: [_candidate("cron")]
    )

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=1,
    )
    payload = out[0].to_dict()
    assert payload["family_candidate"]["name"] == "cron"
    assert payload["mutation"]["o"] == "persistence"
    assert payload["lethal"] is True
    # path_expand no longer fabricates a bypass verdict — the
    # runner's per-entry audit hook overwrites this.  Until then it
    # is None.
    assert payload["bypassed"] is None
    assert isinstance(payload["tool_chain"], list)


def test_path_expand_technique_summary_propagates_across_rounds(monkeypatch):
    """A2: round 2/3 brainstorm with the round-1 winner's technique label.

    Round 1 picks ``cron``; the winning candidate's identity must land
    on ``mutation.metadata['technique_summary']`` so the next round's
    ``_current_technique`` derives the brainstorm seed from *that*
    technique rather than reverting to the parent's objective string.

    Each round's ``family_enumerate`` returns a fresh single candidate
    so we can pin which candidate is "the winner" without ambiguity.
    """
    parent = _parent()
    # Strip the seeded technique_summary so we can assert it gets set
    # explicitly by path_expand rather than carried over.
    parent.metadata.pop("technique_summary", None)

    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    seen_techniques: list[str] = []
    seen_parents: list[SkillVariant] = []

    # Round-specific candidates so we can verify each round's brainstorm
    # is keyed on the previous round's winner, not the original parent.
    round_candidates = [
        [_candidate("cron")],
        [_candidate("systemd")],
        [_candidate("launchd")],
    ]

    def _fe_stub(current_parent, current_technique, *args, **kwargs):
        seen_parents.append(current_parent)
        seen_techniques.append(current_technique)
        return round_candidates[len(seen_techniques) - 1]

    monkeypatch.setattr(pe, "family_enumerate", _fe_stub)

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=3,
        top_k=1,
    )

    assert len(out) == 3
    # Round 1 brainstorms from the parent's objective (no prior winner).
    assert seen_techniques[0] == parent.o.value
    # Round 2 brainstorms from round-1's winner: "cron: cron mechanism".
    assert "cron" in seen_techniques[1]
    assert "mechanism" in seen_techniques[1]
    # Round 3 brainstorms from round-2's winner.
    assert "systemd" in seen_techniques[2]
    # The promoted mutation carries the technique label so log readers
    # can reconstruct the family lineage post-hoc.
    assert out[0].mutation.metadata.get("technique_summary", "").startswith("cron")
    assert out[1].mutation.metadata.get("technique_summary", "").startswith("systemd")
    assert out[2].mutation.metadata.get("technique_summary", "").startswith("launchd")
    # Each round received a different parent — round 2's parent is the
    # round-1 mutation (carrying the cron technique_summary).
    assert seen_parents[0] is parent
    assert seen_parents[1] is not parent
    assert seen_parents[1].metadata["technique_summary"].startswith("cron")
    assert seen_parents[2].metadata["technique_summary"].startswith("systemd")


def test_path_expand_forwards_family_score_weights_from_ablation(monkeypatch):
    """B4: ``ablation['family_score_weights']`` reaches family_enumerate."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    seen_kwargs: list[dict] = []

    def _fe_stub(current_parent, current_technique, *args, **kwargs):
        seen_kwargs.append(kwargs)
        return [_candidate("cron")]

    monkeypatch.setattr(pe, "family_enumerate", _fe_stub)

    pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=1,
        run_config=RunConfig.from_ablation({"family_score_weights": (1.0, 0.0, 0.0)}),
    )

    assert len(seen_kwargs) == 1
    assert seen_kwargs[0]["score_weights"] == (1.0, 0.0, 0.0)


def test_path_expand_default_score_weights_passes_none(monkeypatch):
    """No ablation override ⇒ ``score_weights=None`` (family_enumerate uses default)."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    seen_kwargs: list[dict] = []

    def _fe_stub(current_parent, current_technique, *args, **kwargs):
        seen_kwargs.append(kwargs)
        return [_candidate("cron")]

    monkeypatch.setattr(pe, "family_enumerate", _fe_stub)

    pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=1,
    )

    assert seen_kwargs[0]["score_weights"] is None


def test_path_expand_default_breaks_after_first_lethal(monkeypatch):
    """C9: default behavior is early-stop on first lethal hit per round."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    monkeypatch.setattr(
        pe,
        "family_enumerate",
        lambda *a, **kw: [
            _candidate("cron"),
            _candidate("systemd"),
            _candidate("bashrc"),
        ],
    )

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=3,
        # No ablation override → default early-stop.
    )

    assert len(out) == 1
    # Only the first candidate (cron) was executed.
    assert out[0].family_candidate["name"] == "cron"
    # agent_run + sandbox each invoked exactly once (not three times).
    assert agent_run.call_count == 1
    assert sandbox.call_count == 1


def test_path_expand_does_not_break_on_non_lethal_candidates(monkeypatch):
    """C9: non-lethal candidates don't trigger early-stop; loop continues."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    # First two non-lethal, third lethal.
    oracle_calls = []
    def _oracle(mutation, hp, o):
        oracle_calls.append(mutation)
        return len(oracle_calls) >= 3  # third call returns True
    oracle = MagicMock(side_effect=_oracle)

    monkeypatch.setattr(
        pe,
        "family_enumerate",
        lambda *a, **kw: [
            _candidate("a"),
            _candidate("b"),
            _candidate("c"),
        ],
    )

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=3,
    )

    # All three candidates were tried before we saw a lethal hit.
    assert len(out) == 3
    assert sandbox.call_count == 3
    # Only the third was lethal, so it's the only one that would have
    # been promoted; we then break (no more candidates anyway).
    lethal_results = [r for r in out if r.lethal]
    assert len(lethal_results) == 1


def test_path_expand_threads_tried_names_across_rounds(monkeypatch):
    """B6: round 2 brainstorm receives round 1's candidate names as tried_names."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    seen_tried_names: list[set] = []
    round_candidates = [
        [_candidate("cron"), _candidate("systemd")],
        [_candidate("launchd")],
    ]

    def _fe_stub(current_parent, current_technique, *args, **kwargs):
        # Snapshot the tried_names argument as a frozen set.
        tn = kwargs.get("tried_names") or set()
        seen_tried_names.append(set(tn))
        return round_candidates[len(seen_tried_names) - 1]

    monkeypatch.setattr(pe, "family_enumerate", _fe_stub)

    pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=2,
        top_k=2,
    )

    # Round 1 tried_names is empty (first round, nothing tried yet).
    assert seen_tried_names[0] == set()
    # Round 2 tried_names contains BOTH round-1 candidates (we accumulate
    # the full top-k regardless of which one bypassed).
    assert seen_tried_names[1] == {"cron", "systemd"}


def test_path_expand_threads_blue_findings_from_metadata(monkeypatch):
    """B5: ``parent.metadata['audit_findings']`` reaches family_enumerate."""
    parent = _parent()
    # Stash a finding-like object on parent.metadata.
    finding = SimpleNamespace(
        reason_code="autonomy_encouragement",
        metadata={"red_flags": ["sudo"]},
        evidence=SimpleNamespace(snippet=""),
    )
    parent.metadata["audit_findings"] = [finding]

    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    seen_blue_findings: list = []

    def _fe_stub(current_parent, current_technique, *args, **kwargs):
        seen_blue_findings.append(kwargs.get("blue_findings"))
        return [_candidate("cron")]

    monkeypatch.setattr(pe, "family_enumerate", _fe_stub)

    pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=1,
    )

    assert len(seen_blue_findings) == 1
    assert seen_blue_findings[0] == [finding]


def test_path_expand_passes_cached_skill_texts(monkeypatch):
    """C8: ``cached_skill_texts`` is computed once and threaded into family_enumerate."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    cached_texts_received: list = []

    def _fe_stub(current_parent, current_technique, *args, **kwargs):
        cached_texts_received.append(kwargs.get("cached_skill_texts"))
        return [_candidate("cron")]

    monkeypatch.setattr(pe, "family_enumerate", _fe_stub)

    pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=2,
        top_k=1,
    )

    # Both rounds get the same cached_skill_texts (a list, possibly empty
    # if no SKILL.md fixtures exist on this machine — what matters is
    # that the value is identical across rounds, i.e., not None and
    # not regenerated per round).
    assert len(cached_texts_received) == 2
    assert cached_texts_received[0] is not None
    assert cached_texts_received[1] is not None
    # Same content (can be the same list object since path_expand
    # computes it once outside the loop).
    assert cached_texts_received[0] == cached_texts_received[1]


def test_path_expand_exhaustive_disables_early_stop(monkeypatch):
    """C9: ``ablation['path_expand_exhaustive']`` runs every top-k candidate."""
    parent = _parent()
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    oracle = MagicMock(return_value=True)

    monkeypatch.setattr(
        pe,
        "family_enumerate",
        lambda *a, **kw: [_candidate("a"), _candidate("b"), _candidate("c")],
    )

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=1,
        top_k=3,
        run_config=RunConfig.from_ablation({"path_expand_exhaustive": True}),
    )

    assert len(out) == 3
    assert sandbox.call_count == 3


def test_path_expand_technique_summary_skipped_when_not_lethal(monkeypatch):
    """No lethal hit ⇒ technique_summary stays on the original parent."""
    parent = _parent()
    original_summary = parent.metadata["technique_summary"]
    agent_run = _make_agent_run(parent)
    sandbox = _make_sandbox()
    # Oracle declines every variant — best_next never gets set.
    oracle = MagicMock(return_value=False)

    monkeypatch.setattr(
        pe, "family_enumerate",
        lambda *a, **kw: [_candidate("cron")],
    )

    out = pe.path_expand(
        parent,
        _repo(),
        agent_run=agent_run,
        run_in_sandbox=sandbox,
        oracle_lethal=oracle,
        attack_log_path=None,
        max_rounds=2,
        top_k=1,
    )

    assert len(out) == 2
    # Parent metadata was not mutated (no winner promoted onto it).
    assert parent.metadata["technique_summary"] == original_summary
    # The result mutations also do NOT carry a technique_summary update,
    # because non-lethal candidates are never promoted.
    for r in out:
        # Either the original metadata flows through unchanged from the
        # parent clone, or no key is present — but it must NOT have been
        # rewritten to the candidate name.
        assert not r.mutation.metadata.get(
            "technique_summary", ""
        ).startswith("cron:")
