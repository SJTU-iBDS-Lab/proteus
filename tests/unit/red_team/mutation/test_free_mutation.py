"""Verify the new ``free_mutation`` search-space level.

Phase-1 discovery space: (o, tau) frozen, everything else open, no
"≥1 axis change" requirement. Used by the runner for pre-first-success
rounds; horizontal expansion still uses strict path_expansion /
surface_expansion.
"""
from __future__ import annotations

import pytest

from core.schemas import AttackObjective, Channel, SkillVariant, Topology
from red_team.mutation.search_space import InvariantViolation, SearchSpace


def _parent() -> SkillVariant:
    return SkillVariant(
        o=AttackObjective.DATA_EXFILTRATION,
        tau=Topology.PAIR,
        c={"entry_script": None, "scripts": {}, "declared_apis": [], "deps": []},
        kappa=[Channel.FILESYSTEM],
        d="baseline body",
        poison_index=-1,
        chain=["ftp-client", "task-manager"],
        metadata={"seed_id": "t"},
    )


def test_free_mutation_in_valid_levels():
    # Sanity: the new level name is recognised.
    space = SearchSpace("free_mutation")
    assert space.level == "free_mutation"


def test_free_mutation_mask_exposes_all_non_o_tau():
    mask = SearchSpace("free_mutation").mask()
    assert "o" in mask["frozen"]
    assert "tau" in mask["frozen"]
    for fld in ("c", "kappa", "d", "chain"):
        assert fld in mask["mutable"], f"expected {fld} mutable, got {mask}"


def test_free_mutation_accepts_d_only_change():
    """Unlike strict path_expansion, free_mutation accepts d-only (no ≥1 axis rule)."""
    space = SearchSpace("free_mutation")
    p = _parent()
    child = SkillVariant(
        o=p.o, tau=p.tau, c=dict(p.c), kappa=list(p.kappa),
        d="reworded only", poison_index=p.poison_index,
        chain=list(p.chain), metadata=dict(p.metadata),
    )
    space.validate(p, child)  # no raise


def test_free_mutation_accepts_script_change():
    space = SearchSpace("free_mutation")
    p = _parent()
    new_c = dict(p.c)
    new_c["scripts"] = {"init.sh": "#!/bin/bash\necho ok"}
    child = SkillVariant(
        o=p.o, tau=p.tau, c=new_c, kappa=list(p.kappa),
        d=p.d, poison_index=p.poison_index,
        chain=list(p.chain), metadata=dict(p.metadata),
    )
    space.validate(p, child)


def test_free_mutation_accepts_identity_child():
    """Even exact duplicates pass — process invariant is relaxed."""
    space = SearchSpace("free_mutation")
    p = _parent()
    # Rebuild an identical child to avoid shared-reference equality quirks.
    child = SkillVariant(
        o=p.o, tau=p.tau, c=dict(p.c), kappa=list(p.kappa),
        d=p.d, poison_index=p.poison_index,
        chain=list(p.chain), metadata=dict(p.metadata),
    )
    space.validate(p, child)


def test_free_mutation_rejects_objective_change():
    space = SearchSpace("free_mutation")
    p = _parent()
    child = SkillVariant(
        o=AttackObjective.PERSISTENCE, tau=p.tau, c=dict(p.c),
        kappa=list(p.kappa), d=p.d, poison_index=p.poison_index,
        chain=list(p.chain), metadata=dict(p.metadata),
    )
    with pytest.raises(InvariantViolation, match=r"free_mutation keeps the declared objective"):
        space.validate(p, child)


def test_free_mutation_rejects_topology_change():
    space = SearchSpace("free_mutation")
    p = _parent()
    child = SkillVariant(
        o=p.o, tau=Topology.SINGLE, c=dict(p.c),
        kappa=list(p.kappa), d=p.d, poison_index=p.poison_index,
        chain=["ftp-client"], metadata=dict(p.metadata),
    )
    with pytest.raises(InvariantViolation, match=r"free_mutation keeps the topology"):
        space.validate(p, child)


def test_strict_path_expansion_still_requires_axis_change():
    """Regression check: my Bug #3 fix on path_expansion is preserved."""
    space = SearchSpace("path_expansion")
    p = _parent()
    child = SkillVariant(
        o=p.o, tau=p.tau, c=dict(p.c), kappa=list(p.kappa),
        d="reword", poison_index=p.poison_index,
        chain=list(p.chain), metadata=dict(p.metadata),
    )
    with pytest.raises(InvariantViolation, match=r"requires ≥1 axis change"):
        space.validate(p, child)
