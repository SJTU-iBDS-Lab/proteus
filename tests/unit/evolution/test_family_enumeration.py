"""Unit tests for :mod:`red_team.evolution.family_enumeration` (IMPL-10).

Covers the 3-stage pipeline:

* **Brainstorm** — LLM call is mocked out; we assert both the "happy path"
  (≥ 8 candidates, some passing filter) and the JSON-parse-fail path
  (three malformed replies ⇒ ``[]``).
* **Filter**    — benign_cover is driven by a temporary SKILL.md tree,
  history by a tmp_path attack log, kappa_fit (soft ratio in [0,1]) by
  the parent's kappa.
* **Score**     — B4 additive form
  ``w0·log1p(benign_cover) + w1·log1p(history_succ) + w2·kappa_fit``
  with weights default ``(1, 1, 1)``; verified for descending-score
  ordering and the four canonical ablation presets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.evolution import family_enumeration as fe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _parent(kappa=None) -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={"entry_script": "run.sh"},
        kappa=kappa if kappa is not None else [Channel.FILESYSTEM, Channel.SUBPROCESS],
        d="seed doc",
        chain=["skill-alpha"],
        metadata={"motif_summary": "scheduled re-execution"},
    )


def _make_llm_response(payload: object) -> SimpleNamespace:
    """Build a minimal object that looks like an OpenAI chat completion."""
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _candidate_dict(
    name: str,
    keywords: list[str],
    kappa: list[str],
) -> dict[str, object]:
    return {
        "name": name,
        "mechanism": f"{name} mechanism",
        "kappa": kappa,
        "requires": [],
        "benign_use_keywords": keywords,
        "known_canary_overlap": "",
    }


def _write_skills(root: Path, names_to_texts: dict[str, str]) -> Path:
    """Create a ``clawhub/skills/<name>/SKILL.md`` tree under ``root``.

    Returns the ``.../clawhub/skills`` directory so tests can hand it
    straight to ``family_enumerate`` via ``skill_roots``.
    """
    base = root / "clawhub" / "skills"
    for name, text in names_to_texts.items():
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(text, encoding="utf-8")
    return base


# ---------------------------------------------------------------------------
# Brainstorm helpers
# ---------------------------------------------------------------------------


class _StubClient:
    """Stand-in for ``openai.OpenAI`` used by the brainstorm call."""

    def __init__(self, responses: list[object]):
        # ``responses`` holds JSON-serializable payloads *or* raw strings
        # (already-stringified garbage) we want to return in order.
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=self.completions)

    def _create(self, **kwargs):  # noqa: D401 - mimic openai signature
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
            fe, "_make_brainstorm_client", lambda base_url, api_key: stub
        )
        return stub

    return _apply


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_family_enumerate_brainstorm_returns_candidates(tmp_path, patch_client):
    """10 candidates in, 3 selected out, sorted by score descending."""
    # Four of the ten candidates share keywords with the SKILL.md fixtures.
    cands = [
        _candidate_dict("cronjob", ["cron"], ["filesystem"]),
        _candidate_dict("systemd-timer", ["systemd"], ["filesystem"]),
        _candidate_dict("bashrc-hook", ["bashrc"], ["filesystem"]),
        _candidate_dict("profile-d", ["profile"], ["filesystem"]),
        _candidate_dict("empty-1", ["no-match-x"], ["filesystem"]),
        _candidate_dict("empty-2", ["no-match-y"], ["filesystem"]),
        _candidate_dict("empty-3", ["no-match-z"], ["filesystem"]),
        _candidate_dict("empty-4", ["no-match-w"], ["filesystem"]),
        _candidate_dict("empty-5", ["no-match-v"], ["filesystem"]),
        _candidate_dict("empty-6", ["no-match-u"], ["filesystem"]),
    ]
    patch_client([cands])

    data_root = tmp_path / "skills"
    skills_dir = _write_skills(
        data_root,
        {
            "cron-runner": "This skill manages cron schedules for users.",
            "cron-alert": "Cron monitoring with systemd integration.",
            "shell-init": "Appends helpful shortcuts to ~/.bashrc on install.",
            "profile-loader": "Loads environment snippets from profile.d/*.sh.",
            "unrelated": "Just a boring helper with no hooks.",
        },
    )

    out = fe.family_enumerate(
        _parent(),
        current_technique="cron write to /etc/cron.d/",
        attack_log_path=None,
        top_k=3,
        skill_roots=[skills_dir],
    )

    assert len(out) == 3
    # All returned candidates have score > 0.
    assert all(c.score > 0 for c in out)
    # Score ordering is monotonic non-increasing.
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)
    # No empty-* survivors.
    names = {c.name for c in out}
    assert not any(n.startswith("empty-") for n in names)


def test_family_enumerate_all_filtered(tmp_path, patch_client):
    """B4: only zero on EVERY axis (benign + history + kappa) ⇒ dropped.

    Pre-B4 a zero ``benign_cover`` alone was enough to drop a candidate
    (multiplicative collapse).  Under additive scoring, candidates need
    to score zero on all three axes to be filtered — i.e. no benign
    cover, no prior bypass history, AND zero kappa overlap with the
    parent.  Here we hand-build that all-zero case so the test still
    asserts "everything filtered ⇒ empty list".
    """
    # No-overlap kappa: parent has [filesystem, subprocess]; candidates
    # use ["network"] ⇒ kappa_fit_soft = 0.0.
    cands = [
        _candidate_dict(f"z-{i}", [f"token-that-never-appears-{i}"], ["network"])
        for i in range(8)
    ]
    patch_client([cands])

    skills_dir = _write_skills(
        tmp_path / "skills",
        {"bland": "Just a skill doc with no special keywords."},
    )

    out = fe.family_enumerate(
        _parent(),
        current_technique="cron write",
        attack_log_path=None,
        top_k=3,
        skill_roots=[skills_dir],
    )
    assert out == []


def test_family_enumerate_json_parse_fail_after_retries(tmp_path, patch_client):
    """Three malformed replies ⇒ return [] without raising."""
    stub = patch_client(["not json at all", "{still: broken", "<html>oops</html>"])

    skills_dir = _write_skills(tmp_path / "skills", {"x": "anything"})

    out = fe.family_enumerate(
        _parent(),
        current_technique="cron write",
        attack_log_path=None,
        top_k=3,
        skill_roots=[skills_dir],
    )
    assert out == []
    # All three attempts were actually issued.
    assert len(stub.calls) == 3


def test_family_enumerate_kappa_softens_partial_overlap(tmp_path, patch_client):
    """B4: zero-overlap kappa no longer hard-zeros the score.

    Pre-B4 ``kappa_fit`` was a bool; ``False`` zeroed the multiplicative
    score so any out-of-set channel candidate was dropped — even when
    it had strong benign-cover signal.  This contradicted the
    brainstorm prompt's "same OR adjacent channel subset" guidance.

    Under B4 the soft ratio ``|overlap| / |child_kappa|`` puts a
    zero-overlap candidate **below** a perfect-overlap one by exactly
    the kappa weight, but does not eliminate it — letting the LLM's
    adjacent-channel suggestions stay in contention when their other
    signals are strong.
    """
    cands = [
        # Matching kappa — kappa_fit_soft = 1.0
        _candidate_dict("good", ["cron"], ["filesystem"]),
        # No overlap — kappa_fit_soft = 0.0; survives via benign_cover.
        _candidate_dict("bad", ["cron"], ["network"]),
    ]
    patch_client([cands])

    skills_dir = _write_skills(
        tmp_path / "skills", {"a": "cron is here", "b": "cron too"}
    )

    out = fe.family_enumerate(
        _parent(kappa=[Channel.FILESYSTEM]),
        current_technique="cron",
        attack_log_path=None,
        top_k=5,
        skill_roots=[skills_dir],
    )
    names = [c.name for c in out]
    # B4 contract: both survive (additive scoring keeps any candidate
    # with non-zero signal on at least one axis).
    assert "good" in names
    assert "bad" in names
    by_name = {c.name: c for c in out}
    # ``good`` outranks ``bad`` by exactly the kappa-fit weight (default 1.0).
    assert by_name["good"].score > by_name["bad"].score
    # And the pair is the top-2 (single-keyword skills both match "cron",
    # so benign_cover is identical between the two; only kappa differs).
    assert by_name["good"].score - by_name["bad"].score == pytest.approx(1.0)


def test_family_enumerate_history_amplifies_score(tmp_path, patch_client):
    """A single bypassed log row should add log1p(1) ≈ 0.693 to the score.

    Pre-B4 (multiplicative) "multiplied by 2"; post-B4 (additive) it
    *adds* one ``log1p(history_succ)`` term — but the ordering it
    produces is the same: alpha (with bypass history) outranks beta.
    """
    cands = [
        _candidate_dict("alpha", ["cron"], ["filesystem"]),
        _candidate_dict("beta", ["cron"], ["filesystem"]),
    ]
    patch_client([cands])

    # Two SKILL.md matches, so benign_cover=2 for both candidates.
    skills_dir = _write_skills(
        tmp_path / "skills",
        {
            "s1": "Uses cron to schedule jobs.",
            "s2": "Cron alternative: systemd.",
        },
    )

    # Seed an attack log with a bypassed row mentioning "alpha".
    log_path = tmp_path / "attack_log.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "round": 0,
            "reason": "alpha technique bypassed detection",
            "bypassed": True,
        }) + "\n")
        # Add two more rows so the BM25 filter (≥ 3 rows) doesn't drop.
        fh.write(json.dumps({
            "round": 1,
            "reason": "unrelated gamma entry",
            "bypassed": False,
        }) + "\n")
        fh.write(json.dumps({
            "round": 2,
            "reason": "unrelated delta entry",
            "bypassed": False,
        }) + "\n")

    out = fe.family_enumerate(
        _parent(),
        current_technique="cron",
        attack_log_path=log_path,
        top_k=2,
        skill_roots=[skills_dir],
    )
    by_name = {c.name: c for c in out}
    # Both pass the filter; alpha should be first due to history amplification.
    assert {"alpha", "beta"} <= set(by_name)
    assert out[0].name == "alpha"
    assert out[0].score >= out[1].score


def test_family_enumerate_wrapped_json_list(tmp_path, patch_client):
    """If the LLM wraps the list in a dict we should still find it."""
    cands = [_candidate_dict("only", ["cron"], ["filesystem"])]
    patch_client([{"candidates": cands}])

    skills_dir = _write_skills(tmp_path / "skills", {"s": "cron schedule helper"})

    out = fe.family_enumerate(
        _parent(),
        current_technique="cron",
        attack_log_path=None,
        top_k=1,
        skill_roots=[skills_dir],
    )
    assert [c.name for c in out] == ["only"]


# ---------------------------------------------------------------------------
# B4 — additive scoring + ablation hyperparam tests
# ---------------------------------------------------------------------------


def test_family_enumerate_zero_benign_still_scores_via_kappa(tmp_path, patch_client):
    """B4 fix: a candidate with no benign cover but matching kappa survives.

    Pre-B4 the multiplicative formula made ``benign_cover=0`` an
    automatic disqualifier — even a candidate with a perfect kappa
    fit and prior bypass history would get score=0.  Under additive
    scoring with default uniform weights, that same candidate scores
    ``log1p(0) + log1p(0) + 1.0 = 1.0`` and makes top-k.  This is the
    exact bug B4 was designed to fix; if this test fails the score
    formula has regressed back to multiplicative.
    """
    cands = [
        # No SKILL.md will mention this token → benign_cover=0.
        _candidate_dict(
            "novel-technique", ["never-appears-in-any-skill"], ["filesystem"]
        ),
    ]
    patch_client([cands])

    skills_dir = _write_skills(
        tmp_path / "skills", {"unrelated": "Some unrelated SKILL.md text."}
    )

    out = fe.family_enumerate(
        _parent(),
        current_technique="some technique",
        attack_log_path=None,
        top_k=3,
        skill_roots=[skills_dir],
    )

    assert len(out) == 1
    assert out[0].name == "novel-technique"
    assert out[0].benign_cover == 0
    # kappa_fit_soft = 1.0 (filesystem ⊆ parent's [filesystem, subprocess]).
    assert out[0].kappa_fit == pytest.approx(1.0)
    # Score = 0 + 0 + 1.0 = 1.0 with uniform default weights.
    assert out[0].score == pytest.approx(1.0)


def test_family_enumerate_score_weights_benign_only(tmp_path, patch_client):
    """`benign_only` preset (1, 0, 0): kappa_fit/history are masked out.

    Verifies that ablation['family_score_weights'] = (1, 0, 0) zeros
    the history and kappa contributions so a candidate with high
    benign cover but zero kappa overlap still ranks highest.
    """
    cands = [
        # benign_cover=2, kappa overlap with parent → fit=1.0
        _candidate_dict("balanced", ["cron"], ["filesystem"]),
        # benign_cover=2, NO kappa overlap → fit=0.0
        _candidate_dict("kappa-mismatch", ["cron"], ["network"]),
        # benign_cover=0, kappa overlap → fit=1.0
        _candidate_dict("kappa-only", ["never"], ["filesystem"]),
    ]
    patch_client([cands])

    skills_dir = _write_skills(
        tmp_path / "skills", {"a": "cron is here", "b": "cron too"}
    )

    out = fe.family_enumerate(
        _parent(kappa=[Channel.FILESYSTEM]),
        current_technique="cron",
        attack_log_path=None,
        top_k=5,
        skill_roots=[skills_dir],
        score_weights=(1.0, 0.0, 0.0),
        # Disable axis-diversity adjustment so this test isolates the
        # score-weight mechanism — otherwise ``balanced`` and
        # ``kappa-only`` (both κ=[filesystem]) get a redundancy penalty
        # while ``kappa-mismatch`` (κ=[network]) does not, and the
        # weights-only assertion below would mis-fire.
        axis_diversity_strength=0.0,
    )

    by_name = {c.name: c for c in out}
    # ``kappa-only`` has benign_cover=0 → score=0 → DROPPED.
    assert "kappa-only" not in by_name
    # The two with benign_cover=2 score equally (kappa weighted 0).
    assert by_name["balanced"].score == pytest.approx(by_name["kappa-mismatch"].score)
    import math
    expected = 1.0 * math.log1p(2)
    assert by_name["balanced"].score == pytest.approx(expected)


def test_family_enumerate_score_weights_kappa_only(tmp_path, patch_client):
    """`kappa_only` preset (0, 0, 1): only kappa fit decides the ordering."""
    cands = [
        _candidate_dict("perfect", ["x"], ["filesystem"]),
        _candidate_dict("partial", ["x"], ["filesystem", "network"]),
        _candidate_dict("none", ["x"], ["network"]),
    ]
    patch_client([cands])

    skills_dir = _write_skills(tmp_path / "skills", {"s": "no keyword match here"})

    out = fe.family_enumerate(
        _parent(kappa=[Channel.FILESYSTEM]),
        current_technique="x",
        attack_log_path=None,
        top_k=5,
        skill_roots=[skills_dir],
        score_weights=(0.0, 0.0, 1.0),
    )

    by_name = {c.name: c for c in out}
    # ``none`` has kappa_fit_soft = 0 → score 0 → dropped.
    assert "none" not in by_name
    assert by_name["perfect"].score == pytest.approx(1.0)   # 1/1
    assert by_name["partial"].score == pytest.approx(0.5)   # 1/2
    assert by_name["perfect"].score > by_name["partial"].score


def test_family_enumerate_default_weights_constant_is_uniform():
    """Module default weights are uniform (1.0, 1.0, 1.0).

    Pinned so the default doesn't silently drift before grid-search
    lands. If you change the default you should also update the
    docstring + ``docs/ablation_configs.md`` and bump the
    ``IMPL-evolution-v15+`` TODO referenced there.
    """
    assert fe._DEFAULT_FAMILY_SCORE_WEIGHTS == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# B5 — blue_findings injection into brainstorm prompt
# ---------------------------------------------------------------------------


def _mk_finding(reason_code="autonomy_encouragement", red_flags=None, snippet=""):
    """Minimal SimpleNamespace finding sufficient for the renderer."""
    return SimpleNamespace(
        reason_code=reason_code,
        metadata={"red_flags": list(red_flags or [])},
        evidence=SimpleNamespace(snippet=snippet),
    )


def test_render_constraints_block_includes_blue_findings():
    """B5: reason_codes + red_flags + snippet tokens all appear in the block."""
    findings = [
        _mk_finding(
            reason_code="autonomy_encouragement",
            red_flags=["sudo", "curl"],
            snippet="The skill installs cron entries via crontab.",
        ),
        _mk_finding(
            reason_code="suspicious_network_call",
            red_flags=["wget"],
        ),
    ]
    block = fe._render_constraints_block(
        findings, None, axis_diversity_required=False, brainstorm_min=8
    )
    assert "Avoid (blue-team caught these patterns)" in block
    assert "autonomy_encouragement" in block
    assert "suspicious_network_call" in block
    # red_flags rendered.
    assert "sudo" in block
    assert "curl" in block
    assert "wget" in block
    # Snippet token harvest pulled "crontab" / "installs" (≥4 chars).
    assert "crontab" in block
    # And the LLM-facing instruction is present.
    assert "MUST avoid" in block


def test_render_constraints_block_includes_tried_names():
    """B6: tried_names render as 'Already tried' section."""
    block = fe._render_constraints_block(
        None,
        ["cron", "systemd_timer", ""],  # empty string filtered
        axis_diversity_required=False,
        brainstorm_min=8,
    )
    assert "Already tried (do NOT repeat)" in block
    assert "cron" in block
    assert "systemd_timer" in block
    assert "do NOT" in block.lower() or "do not" in block.lower()


def test_render_constraints_block_axis_diversity_for_large_batch():
    """E14: axis-diversity ask rendered when brainstorm_min >= 4."""
    block = fe._render_constraints_block(
        None, None, axis_diversity_required=True, brainstorm_min=8
    )
    assert "Axis-diversity requirement" in block
    assert "different κ" in block
    assert "alt_chain" in block


def test_render_constraints_block_axis_diversity_skipped_for_small_batch():
    """E14: skip the diversity ask when brainstorm_min < 4 (too small)."""
    block = fe._render_constraints_block(
        None, None, axis_diversity_required=True, brainstorm_min=3
    )
    assert "Axis-diversity" not in block


def test_render_constraints_block_empty_when_nothing_requested():
    """All inputs empty / opted out ⇒ empty string (template stable)."""
    assert fe._render_constraints_block(
        None, None, axis_diversity_required=False, brainstorm_min=8
    ) == ""
    # Even with diversity_required=True, brainstorm_min=2 ⇒ skipped.
    assert fe._render_constraints_block(
        None, None, axis_diversity_required=True, brainstorm_min=2
    ) == ""


def test_brainstorm_passes_blue_findings_into_prompt(tmp_path, patch_client):
    """End-to-end: blue_findings reach the live brainstorm prompt."""
    cands = [_candidate_dict("cron", ["cron"], ["filesystem"])]
    stub = patch_client([cands])

    skills_dir = _write_skills(
        tmp_path / "skills", {"a": "cron schedule helper"}
    )

    fe.family_enumerate(
        _parent(),
        current_technique="cron",
        attack_log_path=None,
        top_k=1,
        skill_roots=[skills_dir],
        blue_findings=[
            _mk_finding(reason_code="autonomy_encouragement", red_flags=["sudo"]),
        ],
    )

    # The first chat.completions.create call's user message must include
    # the rendered constraints.
    assert stub.calls, "no LLM call captured"
    user_msg = stub.calls[0]["messages"][0]["content"]
    assert "autonomy_encouragement" in user_msg
    assert "sudo" in user_msg


def test_brainstorm_uses_temperature_0_8(tmp_path, patch_client):
    """E13: brainstorm runs at temperature 0.8 (was 0.3)."""
    cands = [_candidate_dict("cron", ["cron"], ["filesystem"])]
    stub = patch_client([cands])

    skills_dir = _write_skills(
        tmp_path / "skills", {"a": "cron schedule helper"}
    )

    fe.family_enumerate(
        _parent(),
        current_technique="cron",
        attack_log_path=None,
        top_k=1,
        skill_roots=[skills_dir],
    )

    assert stub.calls
    assert stub.calls[0]["temperature"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# B6 — tried_names dedup at the family_enumerate filter
# ---------------------------------------------------------------------------


def test_family_enumerate_drops_tried_names(tmp_path, patch_client):
    """B6: candidates whose names appear in ``tried_names`` are dropped."""
    cands = [
        _candidate_dict("cron", ["cron"], ["filesystem"]),
        _candidate_dict("systemd", ["systemd"], ["filesystem"]),
    ]
    patch_client([cands])

    skills_dir = _write_skills(
        tmp_path / "skills",
        {"a": "cron schedule", "b": "systemd timer setup"},
    )

    out = fe.family_enumerate(
        _parent(),
        current_technique="persistence",
        attack_log_path=None,
        top_k=5,
        skill_roots=[skills_dir],
        tried_names={"cron"},  # blacklist cron
    )
    names = [c.name for c in out]
    assert "cron" not in names
    assert "systemd" in names


# ---------------------------------------------------------------------------
# E14 — axis-diversity scoring adjustment
# ---------------------------------------------------------------------------


def test_axis_diversity_penalizes_redundant_kappa(tmp_path, patch_client):
    """E14: same-κ candidates get redundancy penalty; unique-κ does not.

    Three candidates all with κ=[filesystem] vs two with κ=[network]: the
    filesystem-cluster candidates get a -2*0.5*0.2 = -0.2 penalty; the
    network-cluster gets -1*0.5*0.2 = -0.1; this should be visible in
    the score ordering when other axes tie.
    """
    cands = [
        _candidate_dict("fs-1", ["x"], ["filesystem"]),
        _candidate_dict("fs-2", ["x"], ["filesystem"]),
        _candidate_dict("fs-3", ["x"], ["filesystem"]),
        _candidate_dict("net-1", ["x"], ["network"]),
        _candidate_dict("net-2", ["x"], ["network"]),
    ]
    patch_client([cands])

    skills_dir = _write_skills(
        tmp_path / "skills",
        {"a": "x present", "b": "x mentioned"},
    )

    out = fe.family_enumerate(
        _parent(kappa=[Channel.FILESYSTEM, Channel.NETWORK]),
        current_technique="x",
        attack_log_path=None,
        top_k=10,
        skill_roots=[skills_dir],
    )

    by_name = {c.name: c for c in out}
    # fs-cluster (3 same κ) penalty = (3-1)*0.5*0.2 = 0.2 each
    # net-cluster (2 same κ) penalty = (2-1)*0.5*0.2 = 0.1 each
    assert by_name["net-1"].score > by_name["fs-1"].score
    assert by_name["net-1"].score - by_name["fs-1"].score == pytest.approx(0.1)


def test_axis_diversity_alt_chain_bonus(tmp_path, patch_client):
    """A candidate with non-empty ``alt_chain`` gets a +strength bonus."""
    cands = [
        # Same kappa for both so redundancy cancels out in the comparison.
        {**_candidate_dict("with-alt", ["x"], ["filesystem"]),
         "alt_chain": ["bitwarden-credential"]},
        _candidate_dict("plain", ["x"], ["filesystem"]),
    ]
    patch_client([cands])

    skills_dir = _write_skills(tmp_path / "skills", {"a": "x is here"})

    out = fe.family_enumerate(
        _parent(),
        current_technique="x",
        attack_log_path=None,
        top_k=5,
        skill_roots=[skills_dir],
    )
    by_name = {c.name: c for c in out}
    # alt_chain bonus = strength = 0.2.
    assert by_name["with-alt"].score - by_name["plain"].score == pytest.approx(0.2)


def test_axis_diversity_strength_zero_disables_signal(tmp_path, patch_client):
    """``axis_diversity_strength=0`` ⇒ identical-κ candidates score equally."""
    cands = [
        _candidate_dict("a", ["x"], ["filesystem"]),
        _candidate_dict("b", ["x"], ["filesystem"]),
        _candidate_dict("c", ["x"], ["filesystem"]),
    ]
    patch_client([cands])

    skills_dir = _write_skills(tmp_path / "skills", {"s": "x is here"})

    out = fe.family_enumerate(
        _parent(),
        current_technique="x",
        attack_log_path=None,
        top_k=5,
        skill_roots=[skills_dir],
        axis_diversity_strength=0.0,
    )
    scores = [c.score for c in out]
    # All three have identical inputs; with diversity off they tie.
    assert scores[0] == scores[1] == scores[2]


# ---------------------------------------------------------------------------
# C8 — cached skill-text loader
# ---------------------------------------------------------------------------


def test_iter_benign_skill_texts_cached_is_memoized(tmp_path, monkeypatch):
    """Same root tuple ⇒ disk scan happens once."""
    skills_dir = _write_skills(
        tmp_path / "skills",
        {"alpha": "alpha skill", "beta": "beta skill"},
    )
    fe._cached_skill_texts.cache_clear()
    call_log: list = []

    real_iter = fe._iter_benign_skill_texts

    def _spy(roots):
        call_log.append(list(roots) if roots else None)
        return real_iter(roots)

    monkeypatch.setattr(fe, "_iter_benign_skill_texts", _spy)

    out1 = fe._iter_benign_skill_texts_cached([skills_dir])
    out2 = fe._iter_benign_skill_texts_cached([skills_dir])
    out3 = fe._iter_benign_skill_texts_cached([skills_dir])

    assert out1 == out2 == out3
    assert len(out1) == 2  # both SKILL.md files read
    # Underlying loader hit exactly once across three accessor calls.
    assert len(call_log) == 1


def test_family_enumerate_cached_skill_texts_skips_disk(tmp_path, monkeypatch):
    """Caller-supplied ``cached_skill_texts`` short-circuits the loader."""
    fe._cached_skill_texts.cache_clear()
    spy_called = []

    def _exploding_loader(_roots):
        spy_called.append(True)
        return ["should_not_be_returned"]

    monkeypatch.setattr(fe, "_iter_benign_skill_texts", _exploding_loader)

    cands = [_candidate_dict("cron", ["explicit"], ["filesystem"])]
    stub = _StubClient([cands])
    monkeypatch.setattr(
        fe, "_make_brainstorm_client", lambda base_url, api_key: stub
    )

    out = fe.family_enumerate(
        _parent(),
        current_technique="cron",
        attack_log_path=None,
        top_k=1,
        cached_skill_texts=["explicit token elsewhere"],  # we control benign cover
    )
    # The disk loader was never invoked because cached_skill_texts was provided.
    assert spy_called == []
    # And benign_cover came from the caller-supplied texts.
    assert len(out) == 1
    assert out[0].benign_cover == 1
