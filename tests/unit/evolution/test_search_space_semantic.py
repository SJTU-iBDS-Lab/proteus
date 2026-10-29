"""Unit tests for the semantic (S1→evasion, S2→path_expansion, S3→surface_expansion)
rename of :class:`red_team.evolution.search_space.SearchSpace`.

These tests lock in the new vocabulary while leaving ``test_search_space.py``
(which continues to exercise the legacy ``S1``/``S2``/``S3`` alias
constructors) untouched.  The legacy alias is kept alive via
``_normalize_level`` + ``DeprecationWarning`` so existing callers keep
working through the transition.
"""
from __future__ import annotations

import copy
import sys
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas.attack_objective import AttackObjective, Channel, Topology
from core.schemas.skill_variant import SkillVariant
from red_team.evolution.search_space import (
    InvariantViolation,
    SearchSpace,
    _LEGACY_LEVELS,
    _VALID_LEVELS,
    shared_motif,
)


def make_parent(**overrides) -> SkillVariant:
    """Canonical parent variant (mirrors ``test_search_space.make_parent``)."""
    base = dict(
        o=AttackObjective.PERSISTENCE,
        tau=Topology.SINGLE,
        c={
            "entry_script": "run.sh",
            "deps": ["requests"],
            "declared_apis": ["http.get"],
        },
        kappa=[Channel.FILESYSTEM, Channel.SUBPROCESS],
        d="# original skill doc",
        chain=["skill-alpha"],
        metadata={
            "tool_chain_prefix": ["read_file", "exec"],
            "file_access": ["/home/user/.bashrc"],
        },
    )
    base.update(overrides)
    return SkillVariant(**base)


def clone_with(parent: SkillVariant, **overrides) -> SkillVariant:
    new_c = copy.deepcopy(parent.c)
    new_kappa = list(parent.kappa)
    new_chain = list(parent.chain)
    new_metadata = copy.deepcopy(parent.metadata)
    kwargs = dict(
        o=parent.o,
        tau=parent.tau,
        c=new_c,
        kappa=new_kappa,
        d=parent.d,
        chain=new_chain,
        metadata=new_metadata,
        poison_index=parent.poison_index,
    )
    kwargs.update(overrides)
    return SkillVariant(**kwargs)


# --------------------------------------------------------------------- registry


class LevelRegistryTests(unittest.TestCase):
    """Lock the semantic level vocabulary at the module level."""

    def test_valid_levels_contains_three_semantic_names(self) -> None:
        self.assertIn("evasion", _VALID_LEVELS)
        self.assertIn("path_expansion", _VALID_LEVELS)
        self.assertIn("surface_expansion", _VALID_LEVELS)

    def test_valid_levels_rejects_legacy_shorthands(self) -> None:
        """``_VALID_LEVELS`` is the authoritative semantic set — legacy
        abbreviations live exclusively in ``_LEGACY_LEVELS``."""
        self.assertNotIn("S1", _VALID_LEVELS)
        self.assertNotIn("S2", _VALID_LEVELS)
        self.assertNotIn("S3", _VALID_LEVELS)

    def test_legacy_map_covers_all_three_abbreviations(self) -> None:
        self.assertEqual(
            _LEGACY_LEVELS,
            {
                "S1": "evasion",
                "S2": "path_expansion",
                "S3": "surface_expansion",
            },
        )


# --------------------------------------------------------------------- masks


class SemanticMaskTests(unittest.TestCase):
    def test_evasion_mask(self) -> None:
        m = SearchSpace("evasion").mask()
        self.assertEqual(m["mutable"], ["d"])
        self.assertEqual(m["frozen"], ["o", "tau", "c", "kappa", "chain"])

    def test_path_expansion_mask(self) -> None:
        m = SearchSpace("path_expansion").mask()
        self.assertEqual(m["mutable"], ["c", "kappa", "d", "chain"])
        self.assertEqual(m["frozen"], ["o", "tau"])

    def test_surface_expansion_mask(self) -> None:
        m = SearchSpace("surface_expansion").mask()
        self.assertEqual(
            m["mutable"], ["o", "tau", "c", "kappa", "d", "chain"]
        )
        self.assertEqual(m["frozen"], [])

    def test_semantic_and_legacy_masks_match(self) -> None:
        """Each legacy alias must yield the exact same mask as its
        semantic counterpart (after normalization).
        """
        pairs = [
            ("S1", "evasion"),
            ("S2", "path_expansion"),
            ("S3", "surface_expansion"),
        ]
        for legacy, semantic in pairs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                legacy_mask = SearchSpace(legacy).mask()
            semantic_mask = SearchSpace(semantic).mask()
            self.assertEqual(legacy_mask, semantic_mask, msg=legacy)


# --------------------------------------------------------------------- deprecation


class LegacyDeprecationTests(unittest.TestCase):
    def test_s1_construction_warns_deprecation(self) -> None:
        with self.assertWarns(DeprecationWarning):
            SearchSpace("S1")

    def test_s2_construction_warns_deprecation(self) -> None:
        with self.assertWarns(DeprecationWarning):
            SearchSpace("S2")

    def test_s3_construction_warns_deprecation(self) -> None:
        with self.assertWarns(DeprecationWarning):
            SearchSpace("S3")

    def test_legacy_level_is_normalized_on_construct(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            space = SearchSpace("S1")
        self.assertEqual(space.level, "evasion")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            space_s2 = SearchSpace("S2")
        self.assertEqual(space_s2.level, "path_expansion")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            space_s3 = SearchSpace("S3")
        self.assertEqual(space_s3.level, "surface_expansion")

    def test_semantic_construction_does_not_warn(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            SearchSpace("evasion")
            SearchSpace("path_expansion")
            SearchSpace("surface_expansion")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(dep, [])


# --------------------------------------------------------------------- invalid


class SemanticInvalidLevelTests(unittest.TestCase):
    def test_unknown_level_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            SearchSpace("unknown")

    def test_s4_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SearchSpace("S4")


# ----------------------------------------------------------- validate (semantic)


class EvasionValidateTests(unittest.TestCase):
    """Semantic ``evasion`` validate equivalence to legacy ``S1`` behavior."""

    def setUp(self) -> None:
        self.parent = make_parent()
        self.space = SearchSpace("evasion")

    def test_pass_change_only_d(self) -> None:
        child = clone_with(self.parent, d="# mutated doc — add benign polish")
        self.space.validate(self.parent, child)

    def test_fail_change_entry_script(self) -> None:
        mutated_c = copy.deepcopy(self.parent.c)
        mutated_c["entry_script"] = "pwned.sh"
        child = clone_with(self.parent, c=mutated_c)
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)

    def test_fail_change_objective(self) -> None:
        child = clone_with(self.parent, o=AttackObjective.DATA_EXFILTRATION)
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)


class PathExpansionValidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = make_parent()
        self.space = SearchSpace("path_expansion")

    def test_pass_change_c_and_kappa(self) -> None:
        mutated_c = copy.deepcopy(self.parent.c)
        mutated_c["entry_script"] = "alt_entry.sh"
        child = clone_with(
            self.parent,
            c=mutated_c,
            kappa=[Channel.NETWORK],
            d="# new path expansion doc",
        )
        self.space.validate(self.parent, child)

    def test_fail_change_tau(self) -> None:
        child = clone_with(
            self.parent,
            tau=Topology.PAIR,
            chain=["skill-alpha", "skill-beta"],
        )
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)


class SurfaceExpansionValidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = make_parent()
        self.space = SearchSpace("surface_expansion")

    def test_pass_with_shared_motif(self) -> None:
        child = clone_with(
            self.parent,
            o=AttackObjective.SUPPLY_CHAIN_ABUSE,
            tau=Topology.PAIR,
            chain=["skill-alpha", "skill-delta"],
            kappa=[Channel.SUBPROCESS, Channel.NETWORK],
            c={"entry_script": "totally_different.py"},
            metadata={
                "tool_chain_prefix": ["read_file", "exec", "curl"],
                "file_access": ["/tmp/unrelated"],
            },
        )
        self.space.validate(self.parent, child)

    def test_fail_no_motif_overlap(self) -> None:
        child = clone_with(
            self.parent,
            o=AttackObjective.GUIDANCE_INJECTION,
            tau=Topology.PAIR,
            chain=["skill-x", "skill-y"],
            kappa=[Channel.MEMORY],
            c={"entry_script": "brand_new.py"},
            d="# fully unrelated doc",
            metadata={
                "tool_chain_prefix": ["completely_different_tool"],
                "file_access": ["/var/log/unrelated.log"],
            },
        )
        with self.assertRaises(InvariantViolation):
            self.space.validate(self.parent, child)


# ----------------------------------------- cross-check legacy == semantic validate


class LegacySemanticValidateEquivalenceTests(unittest.TestCase):
    """Each legacy alias must validate identically to its semantic peer."""

    def setUp(self) -> None:
        self.parent = make_parent()

    def _quiet_space(self, level: str) -> SearchSpace:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return SearchSpace(level)

    def test_s1_equivalent_to_evasion(self) -> None:
        # Valid-for-evasion child (only d changed).
        good_child = clone_with(self.parent, d="# mutated")
        SearchSpace("evasion").validate(self.parent, good_child)
        self._quiet_space("S1").validate(self.parent, good_child)

        # Bad child (kappa change) — both reject.
        bad_child = clone_with(self.parent, kappa=[Channel.NETWORK])
        with self.assertRaises(InvariantViolation):
            SearchSpace("evasion").validate(self.parent, bad_child)
        with self.assertRaises(InvariantViolation):
            self._quiet_space("S1").validate(self.parent, bad_child)

    def test_s2_equivalent_to_path_expansion(self) -> None:
        mutated_c = copy.deepcopy(self.parent.c)
        mutated_c["entry_script"] = "alt.sh"
        good_child = clone_with(self.parent, c=mutated_c, kappa=[Channel.NETWORK])
        SearchSpace("path_expansion").validate(self.parent, good_child)
        self._quiet_space("S2").validate(self.parent, good_child)

    def test_s3_equivalent_to_surface_expansion(self) -> None:
        child = clone_with(
            self.parent,
            o=AttackObjective.SUPPLY_CHAIN_ABUSE,
            tau=Topology.PAIR,
            chain=["skill-alpha", "skill-delta"],
            kappa=[Channel.SUBPROCESS, Channel.NETWORK],
            c={"entry_script": "other.py"},
            metadata={
                "tool_chain_prefix": ["read_file", "exec", "curl"],
                "file_access": ["/tmp/x"],
            },
        )
        SearchSpace("surface_expansion").validate(self.parent, child)
        self._quiet_space("S3").validate(self.parent, child)


if __name__ == "__main__":
    unittest.main()
