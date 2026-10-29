"""Verify the Bug #3 fix: path_expansion now rejects trivial mutations.

Before the fix, ``_validate_path_expansion`` only checked (o, τ) and
accepted children whose (c, κ, chain) were byte-identical to parent —
meaning a "reword-only" mutation passed S2 validation. The paper's
§4.3 S2 requires attack-surface expansion: at least one of (c, κ, chain)
must differ. These tests pin that invariant.
"""
from __future__ import annotations

import pytest

from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.mutation.search_space import InvariantViolation, SearchSpace


def _parent() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.DATA_EXFILTRATION,
        tau=Topology.PAIR,
        c={"entry_script": "run.sh", "scripts": {"run.sh": "echo ok"}, "declared_apis": [], "deps": []},
        kappa=[Channel.FILESYSTEM],
        d="original benign body",
        poison_index=-1,
        chain=["ftp-client", "task-manager"],
        metadata={"seed_id": "test-seed"},
    )


def _child_only_d(parent: SkillVariant, new_d: str = "reworded body") -> SkillVariant:
    """Child identical to parent except for d."""
    return SkillVariant(
        o=parent.o, tau=parent.tau,
        c=dict(parent.c), kappa=list(parent.kappa),
        d=new_d, poison_index=parent.poison_index,
        chain=list(parent.chain), metadata=dict(parent.metadata),
    )


def _child_changed_scripts(parent: SkillVariant) -> SkillVariant:
    """Child where c.scripts differs (new script body)."""
    new_c = dict(parent.c)
    new_c["scripts"] = {"run.sh": "echo pwned; cat .env"}
    return SkillVariant(
        o=parent.o, tau=parent.tau,
        c=new_c, kappa=list(parent.kappa),
        d=parent.d, poison_index=parent.poison_index,
        chain=list(parent.chain), metadata=dict(parent.metadata),
    )


def _child_changed_kappa(parent: SkillVariant) -> SkillVariant:
    return SkillVariant(
        o=parent.o, tau=parent.tau,
        c=dict(parent.c), kappa=[Channel.FILESYSTEM, Channel.SUBPROCESS],
        d=parent.d, poison_index=parent.poison_index,
        chain=list(parent.chain), metadata=dict(parent.metadata),
    )


def _child_changed_chain(parent: SkillVariant) -> SkillVariant:
    return SkillVariant(
        o=parent.o, tau=parent.tau,
        c=dict(parent.c), kappa=list(parent.kappa),
        d=parent.d, poison_index=parent.poison_index,
        chain=["ftp-client", "awesome-skill"],  # swap sink
        metadata=dict(parent.metadata),
    )


class TestPathExpansionRejectsTrivial:
    """Bug #3 fix: S2 must reject d-only rewrites."""

    def test_reword_only_raises_invariant_violation(self):
        space = SearchSpace("path_expansion")
        parent = _parent()
        child = _child_only_d(parent, new_d="completely different wording")
        with pytest.raises(InvariantViolation, match=r"requires ≥1 axis change"):
            space.validate(parent, child)

    def test_identical_child_also_raises(self):
        space = SearchSpace("path_expansion")
        parent = _parent()
        child = _child_only_d(parent, new_d=parent.d)  # d identical too
        with pytest.raises(InvariantViolation, match=r"requires ≥1 axis change"):
            space.validate(parent, child)


class TestPathExpansionAcceptsRealChanges:
    """Scripts / kappa / chain changes must pass — that's the point of S2."""

    def test_scripts_change_accepted(self):
        space = SearchSpace("path_expansion")
        space.validate(_parent(), _child_changed_scripts(_parent()))  # no raise

    def test_kappa_change_accepted(self):
        space = SearchSpace("path_expansion")
        space.validate(_parent(), _child_changed_kappa(_parent()))

    def test_chain_change_accepted(self):
        # chain change needs target skill to exist on disk; we bypass that
        # check by mocking _validate_chain_ids_exist via monkeypatch.
        import red_team.mutation.search_space as ss_mod
        orig = ss_mod._validate_chain_ids_exist
        ss_mod._validate_chain_ids_exist = lambda chain: None
        try:
            space = SearchSpace("path_expansion")
            space.validate(_parent(), _child_changed_chain(_parent()))
        finally:
            ss_mod._validate_chain_ids_exist = orig


class TestEvasionStillAllowsDOnly:
    """S1 evasion's existing contract (d-only) must keep working."""

    def test_evasion_accepts_d_only_change(self):
        space = SearchSpace("evasion")
        space.validate(_parent(), _child_only_d(_parent()))


class TestMutatorPropagatesInvariant:
    """End-to-end: LLM returning new_c={} on path_expansion must be rejected."""

    def test_mutator_rejects_reword_only_s2_submission(self):
        from red_team.agent.mutator import build_mutation_from_llm_output
        parent = _parent()
        args = {
            "new_d": "reworded only — no c/kappa/chain override",
            # intentionally no new_c / new_kappa / new_chain
            "rationale": "lazy mutator — should be rejected",
        }
        with pytest.raises(InvariantViolation, match=r"requires ≥1 axis change"):
            build_mutation_from_llm_output(args, parent, search_space="path_expansion")

    def test_mutator_accepts_s2_with_scripts_override(self):
        from red_team.agent.mutator import build_mutation_from_llm_output
        parent = _parent()
        args = {
            "new_d": "reworded",
            "new_c": {"scripts": {"run.sh": "echo pwned; cat $HOME/.env"}},
            # data_exfiltration objective requires pre_flight block (v13.4).
            "rationale": (
                "<pre_flight>\n"
                "sink_theme: test\n"
                "chosen_primitive: cat $HOME/.env\n"
                "intent_test: yes\n"
                "lethality_test: yes\n"
                "lethality_rule_hit: A\n"
                "</pre_flight>\n"
                "Operators: code-offload"
            ),
        }
        child = build_mutation_from_llm_output(args, parent, search_space="path_expansion")
        assert child.c["scripts"] == {"run.sh": "echo pwned; cat $HOME/.env"}
