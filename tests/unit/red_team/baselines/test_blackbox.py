"""Unit tests for :class:`BlackboxRewrite` (IMPL-18).

All LLM traffic is mocked — we only assert on prompt construction,
result shape, and statefulness of the scalar memory.
"""
from __future__ import annotations

from typing import Any

import pytest

from core.schemas import (
    AttackObjective,
    Channel,
    Evidence,
    PaperFinding,
    SkillVariant,
    Topology,
)
from red_team.baselines.base import AgentRunResult
from red_team.baselines.blackbox_rewrite import BlackboxRewrite


# ---------------------------------------------------------------- fixtures

def _make_seed() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.DATA_EXFILTRATION,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=[Channel.SUBPROCESS, Channel.NETWORK],
        d="seed SKILL.md body",
        poison_index=0,
        chain=["skill-a"],
        metadata={"seed_id": "seed-04"},
    )


def _make_finding() -> PaperFinding:
    return PaperFinding(
        finding_id="A-0099-SECRET",
        layer="A",
        category="guidance_injection_UNIQUE_TOKEN",
        reason_code="autonomy_encouragement_MARKER",
        confidence=0.91,
        severity="high",
        evidence=Evidence(
            file="path/SECRET_FILE.md",
            line_range=(7, 9),
            snippet="SECRET_EVIDENCE_SNIPPET_XYZZY",
            rule_id="rule_SECRET_RULE",
        ),
        explanation="EXPLAIN_SECRET_PHRASE",
        metadata={"note": "META_SECRET_VALUE"},
    )


def _install_mock_llm(
    baseline: BlackboxRewrite,
    captured: list[str],
    *,
    tokens_in: int = 123,
    tokens_out: int = 45,
) -> None:
    """Replace ``baseline._llm_call`` with an in-memory recorder."""

    def fake_llm_call(
        prompt: str,
        *,
        model: str,
        litellm_url: str,
        api_key: str | None,
    ) -> tuple[str, int, int]:
        captured.append(prompt)
        # Embed a unique marker so we can confirm the mutation used our output.
        return (f"rewritten body v{len(captured)}", tokens_in, tokens_out)

    baseline._llm_call = fake_llm_call  # type: ignore[method-assign]


# -------------------------------------------------------- five-round run

def test_five_rounds_produce_five_distinct_prompts() -> None:
    """Alternating bypass bits across 5 rounds yield 5 distinct prompts.

    Also verifies the expected "was bypassed" / "was not bypassed"
    substrings appear in the right rounds and that basic result-shape
    invariants hold every round.
    """
    baseline = BlackboxRewrite(initial_bypass=None, memory_size=3)
    seed = _make_seed()
    captured: list[str] = []
    _install_mock_llm(baseline, captured)

    # Alternating outcomes: round 0 bypass, round 1 not, round 2 bypass, ...
    outcomes = [True, False, True, False, True]
    results: list[AgentRunResult] = []
    for rnd in range(5):
        result = baseline.agent_run(
            seed,
            findings=[],
            repo=None,
            search_space="evasion",
            round_index=rnd,
        )
        results.append(result)
        baseline.update_outcome(rnd, outcomes[rnd])

    # 5 distinct prompts captured (no caching / deduping).
    assert len(captured) == 5
    assert len(set(captured)) == 5

    # Round 0: no prior memory → "first round" language.
    assert "first round" in captured[0]
    assert "previous variant was bypassed" not in captured[0]
    assert "previous variant was not bypassed" not in captured[0]

    # Round 1: previous bit is round-0's True → "was bypassed".
    assert "previous variant was bypassed." in captured[1]
    # Round 2: previous bit is round-1's False → "was not bypassed".
    assert "previous variant was not bypassed." in captured[2]
    # Round 3: previous bit is round-2's True.
    assert "previous variant was bypassed." in captured[3]
    # Round 4: previous bit is round-3's False.
    assert "previous variant was not bypassed." in captured[4]

    # Shape invariants — every round.
    for result in results:
        assert isinstance(result, AgentRunResult)
        assert result.tool_chain == []
        assert result.refusal is False
        assert result.stop_reason == "blackbox"
        assert result.mutation is not None
        assert result.mutation.o == seed.o
        assert result.cost_usd > 0.0


# ------------------------------------------------------ findings ignored

def test_findings_argument_is_ignored_in_prompt() -> None:
    """Passing PaperFinding records must leak none of their fields."""
    baseline = BlackboxRewrite(initial_bypass=True, memory_size=3)
    seed = _make_seed()
    captured: list[str] = []
    _install_mock_llm(baseline, captured)

    finding = _make_finding()
    baseline.agent_run(
        seed,
        findings=[finding],
        repo=None,
        search_space="evasion",
        round_index=0,
    )

    assert len(captured) == 1
    prompt = captured[0]
    # None of the finding's distinctive fields may appear in the prompt.
    for needle in (
        finding.finding_id,
        finding.category,
        finding.reason_code,
        finding.severity,
        finding.evidence.file,
        finding.evidence.snippet,
        finding.evidence.rule_id,
        finding.explanation,
        "META_SECRET_VALUE",
    ):
        assert needle not in prompt, f"prompt leaked finding field: {needle!r}"


# -------------------------------------------------- memory_size semantics

def test_memory_size_one_only_shows_last_bit() -> None:
    """With ``memory_size=1`` the prompt must not include a history trace."""
    baseline = BlackboxRewrite(initial_bypass=None, memory_size=1)
    seed = _make_seed()
    captured: list[str] = []
    _install_mock_llm(baseline, captured)

    # Feed three rounds worth of outcomes.
    outcomes = [True, False, True]
    for rnd in range(3):
        baseline.agent_run(
            seed,
            findings=[],
            repo=None,
            search_space="evasion",
            round_index=rnd,
        )
        baseline.update_outcome(rnd, outcomes[rnd])

    # No "History:" trace ever, regardless of round.
    for prompt in captured:
        assert "History:" not in prompt

    # Round 0 call: no prior outcome → "first round" language.
    assert "first round" in captured[0]
    # Round 1 call: after update_outcome(0, True) → "was bypassed".
    assert "previous variant was bypassed." in captured[1]
    # Round 2 call: after update_outcome(1, False) → "was not bypassed".
    assert "previous variant was not bypassed." in captured[2]


def test_memory_size_three_includes_history_trace() -> None:
    """With ``memory_size=3`` and >=2 outcomes, a History: line appears."""
    baseline = BlackboxRewrite(initial_bypass=None, memory_size=3)
    seed = _make_seed()
    captured: list[str] = []
    _install_mock_llm(baseline, captured)

    outcomes = [True, False, True]
    for rnd in range(3):
        baseline.agent_run(
            seed,
            findings=[],
            repo=None,
            search_space="evasion",
            round_index=rnd,
        )
        baseline.update_outcome(rnd, outcomes[rnd])

    # Round 2 call (index 2) has two prior outcomes in memory → History: line.
    prompt_r2 = captured[2]
    assert "History:" in prompt_r2
    assert "round 0 bypassed" in prompt_r2
    assert "round 1 not-bypassed" in prompt_r2


# ----------------------------------------------- initial_bypass=None at 0

def test_initial_bypass_none_round_zero_says_first_round() -> None:
    """``initial_bypass=None`` at ``round_index=0`` announces "first round"."""
    baseline = BlackboxRewrite(initial_bypass=None, memory_size=3)
    seed = _make_seed()
    captured: list[str] = []
    _install_mock_llm(baseline, captured)

    result = baseline.agent_run(
        seed,
        findings=[],
        repo=None,
        search_space="evasion",
        round_index=0,
    )
    assert len(captured) == 1
    prompt = captured[0]
    assert "Round 0" in prompt
    # Convention: "first round" language when there is no prior bit.
    assert "first round" in prompt
    assert "no previous" in prompt
    # And neither of the scalar phrases should appear.
    assert "was bypassed" not in prompt
    assert "was not bypassed" not in prompt

    # Result shape still valid on round zero.
    assert result.tool_chain == []
    assert result.refusal is False
    assert result.stop_reason == "blackbox"


def test_initial_bypass_true_round_zero_surfaces_scalar() -> None:
    """``initial_bypass=True`` seeds round-0 with a "was bypassed" line."""
    baseline = BlackboxRewrite(initial_bypass=True, memory_size=3)
    seed = _make_seed()
    captured: list[str] = []
    _install_mock_llm(baseline, captured)

    baseline.agent_run(
        seed,
        findings=[],
        repo=None,
        search_space="evasion",
        round_index=0,
    )
    prompt = captured[0]
    assert "previous variant was bypassed." in prompt
    assert "first round" not in prompt


# ------------------------------------------------- update_outcome stateful

def test_update_outcome_appends_and_respects_memory_size() -> None:
    """The internal deque must be bounded by ``memory_size``."""
    baseline = BlackboxRewrite(initial_bypass=None, memory_size=2)
    baseline.update_outcome(0, True)
    baseline.update_outcome(1, False)
    baseline.update_outcome(2, True)
    snapshot = baseline.memory_snapshot
    assert len(snapshot) == 2
    assert snapshot == [(1, False), (2, True)]


# ----------------------------------------------------- cost + token wiring

def test_cost_reflects_token_counts_from_mock() -> None:
    """``cost_usd`` is computed via ``estimate_cost`` on the mock's tokens."""
    baseline = BlackboxRewrite(initial_bypass=None, memory_size=3)
    seed = _make_seed()
    captured: list[str] = []
    _install_mock_llm(baseline, captured, tokens_in=1000, tokens_out=500)

    result = baseline.agent_run(
        seed,
        findings=[],
        repo=None,
        search_space="evasion",
        round_index=0,
        model="glm-5",
    )
    # 1000 in + 500 out @ glm-5 (0.003 / 0.009 per 1k) with 1.3x safety margin.
    expected = (0.003 + 0.5 * 0.009) * 1.3
    assert result.cost_usd == pytest.approx(expected)
    assert result.tokens_in == 1000
    assert result.tokens_out == 500


# ------------------------------------------------------ input validation

def test_memory_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        BlackboxRewrite(memory_size=0)
    with pytest.raises(ValueError):
        BlackboxRewrite(memory_size=-1)


# -------------------------------------------------------- protocol check

def test_blackbox_matches_redteam_baseline_protocol() -> None:
    """Minimal structural assertion: ``name`` is "blackbox" and callable."""
    baseline = BlackboxRewrite()
    assert baseline.name == "blackbox"
    assert callable(baseline.agent_run)
    assert callable(baseline.update_outcome)


# --------------------------------------------- additional kwargs tolerated

def test_agent_run_tolerates_unknown_kwargs() -> None:
    """Extra kwargs from the driver must not crash (``**_`` swallow)."""
    baseline = BlackboxRewrite(initial_bypass=None, memory_size=3)
    seed = _make_seed()
    captured: list[str] = []
    _install_mock_llm(baseline, captured)

    result: Any = baseline.agent_run(
        seed,
        findings=[],
        repo=None,
        search_space="evasion",
        round_index=0,
        max_tool_calls=10,           # from Part 2 signature — should be ignored
        budget_usd=50.0,             # ditto
        ablation={"disable": "x"},   # ditto
        memory_enabled=True,         # ditto
    )
    assert result.stop_reason == "blackbox"
    assert len(captured) == 1
